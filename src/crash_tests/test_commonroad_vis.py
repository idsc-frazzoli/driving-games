from commonroad.visualization.mp_renderer import MPRenderer


def test_vis():
    rnd = MPRenderer(plot_limits=[-30, 120, -140, 20], figsize=(8, 4.5))