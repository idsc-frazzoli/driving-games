from sim_tests.controllers_tests.test_lqr import test_lqr
from sim_tests.controllers_tests.test_nmpc_full_kin_cont import test_nmpc_full_kin_cont
from sim_tests.controllers_tests.test_nmpc_full_kin_dis import test_nmpc_full_kin_dis
from sim_tests.controllers_tests.test_nmpc_full_kin_dis_analytical import test_nmpc_full_kin_analytical
from sim_tests.controllers_tests.test_nmpc_lat_kin_cont import test_nmpc_lat_kin_cont
from sim_tests.controllers_tests.test_nmpc_lat_kin_dis import test_nmpc_lat_kin_dis
from sim_tests.controllers_tests.test_nmpc_lateral_no_path_var import test_nmpc_lat_cont_analytical
from sim_tests.controllers_tests.test_pure_pursuit import test_pure_pursuit
from sim_tests.controllers_tests.test_stanley import test_stanley
from sim_tests.controllers_tests.controller_scenarios.scenario_to_test import scenarios


for key in scenarios.keys():
    test_lqr(scenarios[key])
    test_nmpc_full_kin_cont(scenarios[key])
    test_nmpc_full_kin_dis(scenarios[key])
    test_nmpc_full_kin_analytical(scenarios[key])
    test_nmpc_lat_kin_cont(scenarios[key])
    test_nmpc_lat_kin_dis(scenarios[key])
    test_nmpc_lat_cont_analytical(scenarios[key])
    test_pure_pursuit(scenarios[key])
    test_stanley(scenarios[key])
