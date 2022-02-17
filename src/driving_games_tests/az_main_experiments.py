from datetime import timedelta
from time import perf_counter

from driving_games import without_compmake, VehicleTrackState
from driving_games.collisions_check import joint_simple_collision_cost
from driving_games.resources import get_poly_occupancy
from driving_games.zoo_games import games_zoo
from driving_games.zoo_solvers import solvers_zoo
from driving_games_tests import logger

if __name__ == "__main__":
    do_games_sets = [
        "4way_int_2p_sets",
        "4way_int_3p_sets",
        # "4way_int_3p_sets",
        # "complex_int_6p_sets"
    ]
    do_solvers_sets = [
        "solver-2-pure-security_mNE-fact-noextra",
        "solver-2-pure-security_mNE-naive-noextra",
        # "solver-1-pure-security_mNE-fact",
        # "solver-1-finite_mix-mix_mNE-fact",
        # "solver-1-finite_mix-security_mNE-naive",
        # "solver-1-finite_mix-security_mNE-fact",
    ]
    games_sets = {k: games_zoo[k] for k in do_games_sets}
    solvers_sets = {k: solvers_zoo[k] for k in do_solvers_sets}

    # Note. This is not reliable for timing as the order affects the caching.
    logger.info("Starting experiments")
    tic = perf_counter()
    res = without_compmake(games_sets, solvers_sets)
    toc = timedelta(seconds=perf_counter() - tic)
    logger.info(f"Running all experiment took: {toc}")
    print(joint_simple_collision_cost.cache_info())
    print(VehicleTrackState.to_global_pose.cache_info())
    print(get_poly_occupancy.cache_info())
