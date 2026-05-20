from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared import ctp_log  # noqa: E402


def test_ensure_ctp_logging_uses_run_scoped_log_file(monkeypatch, tmp_path):
    importlib.reload(ctp_log)

    logger = logging.getLogger("ctp")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    monkeypatch.setattr(ctp_log, "_ROOT", tmp_path)
    monkeypatch.setattr(ctp_log, "_RUN_STAMP", "20260416_220332", raising=False)
    monkeypatch.setattr(ctp_log, "_ctp_logger_configured", False)

    ctp_log.ensure_ctp_logging()

    log_path = ctp_log.get_log_path()
    assert log_path == tmp_path / "logs" / "ctp_20260416_220332.log"
    assert log_path.name != "ctp.log"
    assert logger.handlers[0].baseFilename == str(log_path)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
