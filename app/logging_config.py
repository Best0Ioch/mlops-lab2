"""Структуроване (JSON) логування для ML API."""
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(level: int = logging.INFO) -> None:
    """Налаштовує root logger на JSON-формат у stdout."""
    logger = logging.getLogger()
    logger.setLevel(level)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)