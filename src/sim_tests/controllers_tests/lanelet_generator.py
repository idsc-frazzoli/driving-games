import math
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from dg_commons.planning.lanes import DgLanelet
import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle
from typing import Optional
from geometry import translation_angle_from_SE2


class LaneletGenerator:
    def __init__(self, lanelet_net: LaneletNetwork, width: Optional[float] = None):
        self.lanelet_net: LaneletNetwork = lanelet_net
        self.width: Optional[float] = width

    def lanelet_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle, lanelet_id: int) -> Lanelet:
        initial_state = dyn_obs.initial_state.position
        end_state = dyn_obs.prediction.trajectory.state_list[-1].position

        initial_ids = self.lanelet_net.find_lanelet_by_position([initial_state])[0]
        final_ids = self.lanelet_net.find_lanelet_by_position([end_state])[0]
        return_lanelet: Optional[Lanelet] = None

        for initial_id in initial_ids:
            lanelet = self.lanelet_net.find_lanelet_by_id(initial_id)
            merged_lanelets, merged_ids = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet,
                                                                                                  self.lanelet_net)
            for i, merged_id in enumerate(merged_ids):
                if any(item in merged_id for item in final_ids):
                    return_lanelet = merged_lanelets[i]
                    break
            if return_lanelet is not None:
                break

        if return_lanelet is None:
            if self.width is None:
                self._compute_width()
            return_lanelet = self._lanelet_from_scratch(dyn_obs, lanelet_id)

        return return_lanelet

    def _lanelet_from_scratch(self, dyn_obs: DynamicObstacle, lanelet_id):
        center_points = []
        for state in dyn_obs.prediction.trajectory.state_list:
            center_points.append(state.position)

        return self.lanelet_from_trajectory(center_points, lanelet_id)

    def lanelet_from_trajectory(self, center_points, lanelet_id: int):
        if self.width is None:
            self._compute_width()

        left, right = self._left_right_from_trajectory(center_points)
        add_center, add_left, add_right = self._additional_trajectory_steps(center_points[-1])

        center_points = np.concatenate((center_points, add_center))
        left = np.concatenate((left, add_left))
        right = np.concatenate((right, add_right))

        return Lanelet(left_vertices=left, right_vertices=right, center_vertices=center_points, lanelet_id=lanelet_id)

    def _compute_width(self):
        res = 0
        counter = 0
        for lanelet in self.lanelet_net.lanelets:
            weight, _ = lanelet.right_vertices.shape
            res += np.average(np.linalg.norm(lanelet.right_vertices - lanelet.left_vertices, axis=1)) * weight
            counter += weight

        self.width = res/counter

    def _left_right_from_trajectory(self, center_points):
        left = []
        right = []

        def update_left_right(point_1: np.ndarray, point_2: np.ndarray):
            delta = point_2 - point_1
            delta /= np.linalg.norm(delta)
            delta *= self.width / 2
            left.append(np.array([-delta[1], delta[0]]))
            right.append(np.array([delta[1], -delta[0]]))

        update_left_right(center_points[0], center_points[1])
        for i, _ in enumerate(center_points):
            if i != 0 and i != (len(center_points) - 1):
                update_left_right(center_points[i - 1], center_points[i + 1])

        update_left_right(center_points[-2], center_points[-1])

        return np.array(left), np.array(right)

    def _additional_trajectory_steps(self, end_position: np.ndarray):
        current_lanelet_id = self.lanelet_net.find_lanelet_by_position([end_position])[0][0]
        current_lanelet = self.lanelet_net.find_lanelet_by_id(current_lanelet_id)
        merged_lanelets, _ = current_lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet,
                                                                                             self.lanelet_net)
        merged_lanelet: Lanelet = merged_lanelets[0]
        center_points = merged_lanelet.center_vertices.tolist()

        dg_lanelet = DgLanelet.from_commonroad_lanelet(merged_lanelet)
        beta, _ = dg_lanelet.find_along_lane_closest_point(end_position)
        first_center, _ = translation_angle_from_SE2(dg_lanelet.center_point(beta))

        center_points = center_points[math.ceil(beta):]
        center_points = [first_center.tolist()] + center_points
        center_points = np.array(center_points)

        left, right = self._left_right_from_trajectory(center_points)
        return center_points, left, right
