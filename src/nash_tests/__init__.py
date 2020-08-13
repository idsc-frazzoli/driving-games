import logging

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger("").addHandler(console)