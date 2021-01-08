import gym_duckietown
from gym_duckietown.simulator import Simulator

def test_sim():
    env = Simulator(
            seed=123, # random seed
            map_name="4way",
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
            frame_rate=30 # todo maybe useful
        )
    while True:
        action = [0.1,0.1]
        observation, reward, done, misc = env.step(action)
        print(misc)
        #env.render(mode="free_cam")
        env.render(mode="top_down")
        if done:
            # break
            env.reset()
