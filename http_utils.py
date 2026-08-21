# -*- coding: utf-8 -*-
"""HTTP 工具：Session 管理、熔断器、指数退避"""

import requests
import time
import threading
from enum import Enum
from typing import Optional
from datetime import datetime, timedelta
from config import settings


HTTP_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
_thread_local = threading.local()


def get_http_session() -> requests.Session:
    """每个线程复用自己的 Session"""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HTTP_HEADERS)
        _thread_local.session = session
    return session


class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开试探


class CircuitBreaker:
    """熔断器：防止级联故障"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: int = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold or settings.circuit_failure_threshold
        self.recovery_timeout = recovery_timeout or settings.circuit_recovery_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN

    @property
    def status(self) -> str:
        return f"{self.name}: {self.state.value} (failures={self.failures})"


# 全局熔断器实例
_api_breaker = CircuitBreaker("api_primary")
_api_alt_breaker = CircuitBreaker("api_alt")


def safe_json_get(url: str, timeout: tuple = None, retries: int = None) -> Optional[dict]:
    """带熔断器和指数退避的安全请求"""
    timeout = timeout or (settings.api_timeout_connect, settings.api_timeout_read)
    retries = retries or settings.api_retries
    session = get_http_session()

    for attempt in range(retries + 1):
        if not _api_breaker.can_execute():
            return None

        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            _api_breaker.record_success()
            return data
        except (requests.RequestException, ValueError) as e:
            _api_breaker.record_failure()
            if attempt >= retries:
                return None
            time.sleep(0.35 * (2 ** attempt))
    return None



def parse_api_time(s):
    """解析接口时间"""
    from datetime import datetime
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        ts = float(s)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts)
        except Exception:
            return None
    text = str(s).strip()
    if text.isdigit():
        ts = float(text)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts)
        except Exception:
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except Exception:
            continue
    return None
