from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons import PlayerName, DgSampledSequence
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt
from geometry import SE2_from_xytheta, SE2value
from homotopies.MILP.utils.visualization import visualize_traj
from homotopies.MILP.utils.prediction import traj_from_commonroad




#visualize vehicles
scenario_name = "USA_Lanker-2_10_T-1"
scenario_dir = "scenarios"
scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenario_dir)
fig = plt.figure(figsize=(25, 10))
rnd = MPRenderer()
# fig.show()
for item in scenario.dynamic_obstacles:
    print(item.obstacle_id)
    scenario.lanelet_network.draw(rnd, draw_params={"traffic_light": {"draw_traffic_lights": False}})
    item.draw(rnd)
    rnd.render()

#extract trajectory from commonroad states
scenario_name = "USA_Lanker-2_10_T-1"
traj = traj_from_commonroad(scenario_name, 2653, (0.,50.))
player = PlayerName("test")
ax = plt.gca()
visualize_traj(traj, player, ax)
plt.show()
print(traj.values)

