from zuper_commons.types import ZValueError

from games_scripts import dg_demo


def test_run1():
    args = ["-o", "out/tests/test_run1"]
    ret = dg_demo(args=args, sys_exit=False)
    if ret:
        raise ZValueError(ret=ret)
