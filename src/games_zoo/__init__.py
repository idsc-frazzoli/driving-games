from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .solvers import *
from .zoo import *
from .demo import *


# Give each symbol this module name
for a in list(globals()):
    v = globals()[a]
    if hasattr(v, "__module__") and v.__module__.startswith(__name__):
        logger.info(f"{a} {v.__module__} -> {__name__}")
        setattr(v, "__module__", __name__)
