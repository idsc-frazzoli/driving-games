from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .preferences_base import *
from .preferences_lexicographic import *
from .preferences_sets import *
from .preferences_conversions import *
from .preferences_scalar import *
from .preferences_product import *
from .operations import *

# # Give each symbol this module name
# for a in list(globals()):
#     v = globals()[a]
#     if hasattr(v, '__module__') and v.__module__.startswith(__name__):
#         logger.info(f'{a} {v.__module__} -> {__name__}')
#         setattr(v, '__module__', __name__)
