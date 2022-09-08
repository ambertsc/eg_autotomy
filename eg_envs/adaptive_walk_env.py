import os

import numpy as np

from gym import spaces

from evogym import EvoWorld
from evogym.envs import EvoGymBase


class AdaptiveWalkEnvClass(EvoGymBase):
    
    def __init__(self, body, connections=None):

        this_filepath = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
        filepath = os.path.join(this_filepath,  "world_data", "adaptive_walk_env.json")
        self.world = EvoWorld.from_json(filepath)
            
        self.add_robot(body, connections)
        
        super().__init__(self.world)
        self.setup_action_space()

        self.default_viewer.track_objects("robot") 


    def get_obs(self):

        obs = np.concatenate((\
                self.get_vel_com_obs("robot"),\
                self.get_relative_pos_obs("robot")))

        return obs

    def add_robot(self, body, connections):

        self.robot_body = 1.0 * body
        self.robot_body_elements = self.robot_body.shape[0] * self.robot_body.shape[1]

        self.world.add_from_array("robot", body, 8, 8, connections=connections) 

    def setup_action_space(self):

        num_actuators = self.get_actuator_indices("robot").size 
        obs_size = self.get_obs().size

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators, ), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape = (obs_size,), dtype=np.float)

    def remove_robot(self, name="robot"):
        
        self.world.remove_object(name)

    def step(self, action):

        obs, reward, done, info  = 0, 0, True, {}

        position_1 = self.object_pos_at_time(self.get_time(), "robot")

        done = super().step({"robot": action})

        position_2 = self.object_pos_at_time(self.get_time(), "robot")

        center_of_mass_1 = np.mean(position_1,1)
        center_of_mass_2 = np.mean(position_2,1)

        reward = center_of_mass_2[0] - center_of_mass_1[0]

        if done:
            print("***UNSTABLE SIMULATION***")
            print("   terminating with penalty -3 ")

            reward -= 3

        elif center_of_mass_2[0] >= 120:
            done = True
            reward += 1

        obs = self.get_obs()
        
        return obs, reward, done, info

    def reset(self):

        super().reset()

        obs = self.get_obs()
        
        return obs

