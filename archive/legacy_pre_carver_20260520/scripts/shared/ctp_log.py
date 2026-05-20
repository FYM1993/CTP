"""
CTP 项目日志

- 默认按进程启动时间写入项目根目录 ``logs/ctp_YYYYMMDD_HHMMSS.log``。
- 盘前 / 筛选 / Phase2 长篇分析 / 盘中仪表盘 → 使用 ``get_logger(__name__)`` 打日志。
- 终端仅保留用户明确需要的提示（如观望品种反转警报），在 ``daily_workflow.phase_3`` 里用 ``print``。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH: Path | None = None
_ctp_logger_configured = False


def get_log_path() -> Path:
    global _LOG_PATH
    if _LOG_PATH is None:
        _LOG_PATH = _ROOT / "logs" / f"ctp_{_RUN_STAMP}.log"
    return _LOG_PATH


def ensure_ctp_logging() -> None:
    """为 logger ``ctp`` 配置单文件 Handler；子 logger ``ctp.xxx`` 会向上冒泡到该 Handler。"""
    global _ctp_logger_configured
    log = logging.getLogger("ctp")
    if log.handlers:
        _ctp_logger_configured = True
        return
    log_path = get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.propagate = False
    _ctp_logger_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    返回 ``ctp.{name}`` 子 logger；首次调用时初始化文件输出。

    Example
    -------
    >>> from shared.ctp_log import get_logger
    >>> log = get_logger("pre_market")
    >>> log.info("...")
    """
    ensure_ctp_logging()
    return logging.getLogger(f"ctp.{name}")
