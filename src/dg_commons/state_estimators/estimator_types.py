from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
from typing import Union


class Empty:
    pass


Estimators = Union[Empty, ExtendedKalman]
EstimatorsParams = Union[Empty, ExtendedKalmanParam]
