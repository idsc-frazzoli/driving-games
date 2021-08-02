from logging import DEBUG, INFO

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)
logger.setLevel(level=INFO)
