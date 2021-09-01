from dg_commons import DgSampledSequence
from sim.models import kmh2ms

from sim.models.vehicle import VehicleModel, VehicleState
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from dg_commons.controllers.speed import SpeedBehavior, SpeedController, SpeedControllerParam, SpeedBehaviorParam
from sim.models.vehicle import VehicleCommands
import matplotlib.pyplot as plt
from decimal import Decimal


def sim_step_response(model, sp_controller):
    t, sim_step = 0, 0.05
    ref = DgSampledSequence[float](timestamps=[0, 10], values=[kmh2ms(40), kmh2ms(5)])
    times, speeds, accelerations, refs = [], [], [], []
    while t < 20:
        current_state = model.get_state().vx
        speeds.append(current_state)
        times.append(t)

        sp_controller.update_measurement(measurement=current_state)
        sp_controller.update_reference(reference=ref.at_or_previous(t))
        refs.append(ref.at_or_previous(t))
        acc = sp_controller.get_control(t)
        accelerations.append(acc)
        cmds = VehicleCommands(acc=acc, ddelta=0)
        model.update(commands=cmds, dt=Decimal(0.1))
        t += sim_step
        # update observations

    # do plot
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Step Response PID Controller for Vehicle Speed")
    ax1.plot(times, speeds, label="actual speed")
    ax1.plot(times, refs, "r", label="ref. speed")
    ax2.plot(times, accelerations)

    ax1.set(ylabel='Velocity')
    ax2.set(xlabel='Time', ylabel='Acceleration Command')
    ax1.legend()
    plt.savefig("fig")


vehicle_speed_step: float = 5
"""Nominal speed of the vehicle"""
speed_kp: float = 4
"""Propotioanl gain longitudinal speed controller"""
speed_ki: float = 0.005
"""Integral gain longitudinal speed controller"""
speed_kd: float = 0.0001
"""Derivative gain longitudinal speed controller"""

sp_controller_param: SpeedControllerParam = SpeedControllerParam(kP=speed_kp, kI=speed_ki, kD=speed_kd)
sp_controller = SpeedController(sp_controller_param)
sp_controller.update_reference(reference=vehicle_speed_step)
"""Speed Controller"""

"Kinematic Model"
#x_0 = VehicleState(0, 0, 0, 0, 0)
#model = VehicleModel.default_car(x_0)

"Dynamic Model"
x_0 = VehicleStateDyn(0, 0, 0, 0, 0, 0, 0)
model = VehicleModelDyn.default_car(x_0)


sim_step_response(model, sp_controller)


