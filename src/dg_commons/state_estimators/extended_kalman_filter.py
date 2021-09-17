import math
from scipy.integrate import solve_ivp
from sim.models.vehicle import VehicleState, VehicleCommands, VehicleGeometry
from sim.models.vehicle_utils import steering_constraint, VehicleParameters
from sim.models.model_utils import acceleration_constraint
import numpy as np
from dataclasses import dataclass
from typing import Optional

geo = VehicleGeometry.default_car()
params = VehicleParameters.default_car()
n_states = VehicleState.get_n_states()
n_commands = VehicleCommands.get_n_commands()
l = geo.length
lr = geo.lr


@dataclass
class ExtendedKalmenParam:
    actual_model_var: np.ndarray = np.zeros((n_states, n_states))
    """ Actual Modeling variance matrix """
    actual_meas_var: np.ndarray = np.zeros((n_states, n_states))
    """ Actual Measurement variance matrix """
    belief_model_var: np.ndarray = actual_model_var
    """ Belief modeling variance matrix """
    belief_meas_var: np.ndarray = actual_meas_var
    """ Belief measurement variance matrix """
    initial_variance: np.ndarray = actual_meas_var
    """ Initial variance matrix """


class ExtendedKalman:
    def __init__(self, dt, x0=None, params=ExtendedKalmenParam()):
        self.actual_model_noise = params.actual_model_var
        self.actual_meas_noise = params.actual_meas_var
        self.belief_model_noise = params.belief_model_var
        self.belief_meas_noise = params.belief_meas_var
        self.p = params.initial_variance

        self.state = x0
        self.dt = dt

    def update_prediction(self, u_k: Optional[VehicleCommands]):
        if self.state is None or u_k is None:
            return

        self.state, self.p = self.solve_dequation(u_k)

    def update_measurement(self, measurement_k: VehicleState):
        if self.state is None:
            self.state = measurement_k
            self.p = self.belief_meas_noise
            return

        h = self.h(self.state)
        state = self.state.as_ndarray().reshape((n_states, 1))
        # Perturb measurement and reshape
        measurement_k = measurement_k + ExtendedKalman.realization(self.actual_meas_noise)
        meas = measurement_k.as_ndarray().reshape((n_states, 1))
        try:
            helper = np.linalg.inv(np.matmul(np.matmul(h, self.p), h.T) + self.belief_meas_noise)
            k = np.matmul(np.matmul(self.p, h.T), helper)
            state = state + np.matmul(k, (meas - state))
            self.state = VehicleState.from_array(np.matrix.flatten(state))
            self.p = np.matmul(np.eye(n_states)-np.matmul(k, h), self.p)
        except np.linalg.LinAlgError:
            # assert self.state == measurement_k
            pass
        except Exception:
            pass

    def solve_dequation(self, u_k: VehicleCommands):

        def vec_to_mat(v):
            return v.reshape(n_states, n_states)

        def mat_to_vec(mat):
            return np.matrix.flatten(mat)

        def _stateactions_from_array(state_input: np.ndarray) -> [VehicleState, VehicleCommands]:
            state = VehicleState.from_array(state_input[0:n_states])
            actions = VehicleCommands(acc=state_input[VehicleCommands.idx["acc"] + n_states],
                                      ddelta=state_input[VehicleCommands.idx["ddelta"] + n_states])
            return state, actions

        def _dynamics(t, y):
            part1, part2 = y[:(n_states + n_commands)], y[(n_states + n_commands):]
            state0, actions = _stateactions_from_array(state_input=part1)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(VehicleCommands.idx)])

            f = self.f(state0)
            p = vec_to_mat(part2)
            dp = np.matmul(f, p) + np.matmul(p, f.T) + self.belief_model_noise

            return np.concatenate([dx.as_ndarray(), du, mat_to_vec(dp)])

        state_zero = np.concatenate([self.state.as_ndarray(), u_k.as_ndarray(), mat_to_vec(self.p)])
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(self.dt)), y0=state_zero)
        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")

        result_values = result.y[:, -1]
        part1, part2 = result_values[:(n_states + n_commands)], result_values[(n_states + n_commands):]
        new_state, _ = _stateactions_from_array(state_input=part1)
        new_p = vec_to_mat(part2)

        return new_state, new_p

    def f(self, state):
        s_t = math.sin(state.theta)
        c_t = math.cos(state.theta)
        t_d = math.tan(state.delta)
        c_d = math.cos(state.delta)
        v = state.vx
        return np.array([[0, 0, -v*s_t-c_t*t_d*lr*v/l, c_t-t_d*s_t*lr/l, -v*s_t*lr/(l*c_d**2)],
                         [0, 0,  v*c_t-s_t*t_d*lr*v/l, s_t+t_d*c_t*lr/l, v*c_t*lr/(l*c_d**2)],
                         [0, 0, 0, t_d/l, v/(l*c_d**2)],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    def h(self, state):
        return np.eye(5)

    def dynamics(self, x0: VehicleState, u: VehicleCommands) -> VehicleState:
        """ Kinematic bicycle model, returns state derivative for given control inputs """
        noise = ExtendedKalman.realization(self.actual_model_noise)

        vx = x0.vx
        dtheta = vx * math.tan(x0.delta) / geo.length
        vy = dtheta * geo.lr
        costh = math.cos(x0.theta)
        sinth = math.sin(x0.theta)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh

        ddelta = steering_constraint(x0.delta, u.ddelta, params)
        acc = acceleration_constraint(x0.vx, u.acc, params)
        return VehicleState(x=xdot, y=ydot, theta=dtheta, vx=acc, delta=ddelta) + noise

    @staticmethod
    def realization(var: np.ndarray):
        dim = int(var.shape[0])
        return VehicleState.from_array(np.random.multivariate_normal(np.zeros(dim), var))


'''model_noise = 0.1*np.eye(n_states)
meas_noise = 0.1*np.zeros((n_states, n_states))
P0_0 = np.zeros((n_states, n_states))
dt = 0.1
x0_0 = VehicleState.from_array(np.zeros(n_states))
input_k = VehicleCommands.from_array(np.ones(n_commands))
meas_k = VehicleState.from_array(np.ones(n_states))

test = ExtendedKalman(dt, x0_0, params=ExtendedKalmenParam(model_noise, meas_noise))
print(test.state)
print(test.p)
test.update_prediction(input_k)
test.update_measurement(meas_k)
#test.update_measurement(meas_k)
print(test.state)
print(test.p)'''

