# -*- coding: utf-8 -*-
"""数据库层：SQLite + WAL + 批量写入 + 优化索引"""

import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict
from contextlib import contextmanager
from config import settings
from models.types import PredictionRecord


class DatabaseManager:
    """线程安全的数据库管理器"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_queue: List[PredictionRecord] = []
        self._queue_lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self):
        """初始化数据库表和索引"""
        with self._get_conn() as conn:
            # 主表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pkey TEXT NOT NULL,
                    issue TEXT NOT NULL,
                    category TEXT NOT NULL,
                    pattern TEXT DEFAULT '',
                    lean TEXT DEFAULT '',
                    sample INTEGER DEFAULT 0,
                    pct REAL DEFAULT 0,
                    actual TEXT DEFAULT '',
                    result TEXT DEFAULT '待开',
                    created_at TEXT NOT NULL,
                    settle_issue TEXT DEFAULT '',
                    confidence TEXT DEFAULT '低',
                    model_name TEXT DEFAULT '',
                    model_score REAL DEFAULT 0,
                    UNIQUE(pkey, issue, category)
                )
            """)

            # 优化索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_key_issue 
                ON predictions(pkey, issue)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_settle 
                ON predictions(pkey, result, model_name, category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_time 
                ON predictions(pkey, created_at DESC)
            """)

            # 兼容旧数据库
            self._migrate_old_db(conn)
            conn.commit()

    def _migrate_old_db(self, conn: sqlite3.Connection):
        """迁移旧数据库结构"""
        cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        migrations = {
            'confidence': "ALTER TABLE predictions ADD COLUMN confidence TEXT DEFAULT '低'",
            'model_name': "ALTER TABLE predictions ADD COLUMN model_name TEXT DEFAULT ''",
            'model_score': "ALTER TABLE predictions ADD COLUMN model_score REAL DEFAULT 0",
        }
        for col, sql in migrations.items():
            if col not in cols:
                conn.execute(sql)

    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def upsert_prediction(self, record: PredictionRecord):
        """插入或更新预测记录"""
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO predictions(
                    pkey, issue, category, pattern, lean, sample, pct,
                    actual, result, created_at, settle_issue, confidence,
                    model_name, model_score
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pkey, issue, category) DO UPDATE SET
                    pattern=excluded.pattern, lean=excluded.lean,
                    sample=excluded.sample, pct=excluded.pct,
                    confidence=excluded.confidence,
                    model_name=excluded.model_name,
                    model_score=excluded.model_score
            """, record.to_row())

    def batch_upsert(self, records: List[PredictionRecord]):
        """批量写入"""
        if not records:
            return
        with self.transaction() as conn:
            conn.executemany("""
                INSERT INTO predictions(
                    pkey, issue, category, pattern, lean, sample, pct,
                    actual, result, created_at, settle_issue, confidence,
                    model_name, model_score
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pkey, issue, category) DO UPDATE SET
                    pattern=excluded.pattern, lean=excluded.lean,
                    sample=excluded.sample, pct=excluded.pct,
                    confidence=excluded.confidence,
                    model_name=excluded.model_name,
                    model_score=excluded.model_score
            """, [r.to_row() for r in records])

    def queue_write(self, record: PredictionRecord):
        """加入写入队列（异步批量）"""
        with self._queue_lock:
            self._write_queue.append(record)
            if len(self._write_queue) >= settings.db_batch_size:
                self._flush_queue()

    def _flush_queue(self):
        """刷新写入队列"""
        with self._queue_lock:
            if self._write_queue:
                self.batch_upsert(self._write_queue)
                self._write_queue.clear()

    def settle_prediction(self, pid: int, actual: str, result: str, settle_issue: str):
        """结算预测"""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE predictions 
                SET actual=?, result=?, settle_issue=? 
                WHERE id=?
            """, (actual, result, str(settle_issue), int(pid)))

    def load_predictions(
        self, 
        key: str, 
        limit: int = 500,
        category: Optional[str] = None,
        result: Optional[str] = None,
    ) -> List[PredictionRecord]:
        """加载预测记录（带过滤）"""
        conditions = ["pkey = ?"]
        params = [key]

        if category:
            conditions.append("category = ?")
            params.append(category)
        if result:
            conditions.append("result = ?")
            params.append(result)

        sql = f"""
            SELECT id, pkey, issue, category, pattern, lean, sample, pct,
                   actual, result, created_at, settle_issue, confidence,
                   model_name, model_score
            FROM predictions
            WHERE {' AND '.join(conditions)}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(int(limit))

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [PredictionRecord.from_row(tuple(r)) for r in rows]

    def get_model_performance(
        self, 
        key: str, 
        limit: int = 1000,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取模型表现统计"""
        sql = """
            SELECT model_name, category, result, created_at
            FROM predictions
            WHERE pkey = ? AND result IN ('对','错') AND model_name <> ''
        """
        params = [key]
        if category and category != '全部':
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {'model': r[0], 'category': r[1], 'result': r[2], 'time': r[3]}
            for r in rows
        ]

    def get_stats_summary(self, key: str, window: int = 300) -> Dict[str, Any]:
        """获取统计摘要"""
        with self._get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE pkey = ?",
                (key,)
            ).fetchone()[0]

            settled = conn.execute("""
                SELECT category, result, COUNT(*) 
                FROM predictions 
                WHERE pkey = ? AND result IN ('对','错')
                GROUP BY category, result
            """, (key,)).fetchall()

        stats = defaultdict(lambda: {"对": 0, "错": 0})
        for cat, res, cnt in settled:
            stats[cat][res] = cnt

        return {
            "total_records": total,
            "category_stats": dict(stats),
        }

    def clear_predictions(self, key: str):
        """清空彩种预测记录"""
        with self.transaction() as conn:
            conn.execute("DELETE FROM predictions WHERE pkey = ?", (key,))

    def close(self):
        """关闭连接"""
        self._flush_queue()
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# 全局数据库实例
db = DatabaseManager()
