import os

import numpy as np

from gym import spaces

import scipy
from scipy.ndimage import label
from evogym import EvoWorld, sample_robot
from evogym.envs import EvoGymBase
import copy

from eg_auto.helpers import check_connected

class BackAndForthEnvClass(EvoGymBase):
    
    def __init__(self, body=None, connections=None, **kwargs):

        self.max_episode_steps = 2048

        if "use_difficulty" in kwargs.keys():
            self.use_difficulty = kwargs["use_difficulty"]
            self.difficulty_level = 0
            self.max_difficulty = 2
        else:
            self.use_difficulty = 0
            self.difficulty_level = 0
            self.max_difficulty = 2

        this_filepath = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

        if self.use_difficulty and self.difficulty_level == 0: 
            filepath = os.path.join(this_filepath,  "world_data", \
                    "flat_walk_diff1.json")
        else:
            filepath = os.path.join(this_filepath,  "world_data", "flat_walk.json")
        self.world = EvoWorld.from_json(filepath)
            
        if body is None:
            # establish default body plan
            body = np.ones((4,4))
            body[1:-1,1:-1] = 3
            body[1,:] = 4
            body[-1,1:-1] = 0 


        self.mode = np.array([0])

        if "goal" in kwargs.keys():
            self.goal = kwargs["goal"]
            assert len(self.goal) == 2
        else:
            self.goal = [48, 16]
            
        if "allow_autotomy" in kwargs.keys():
            self.allow_autotomy = kwargs["allow_autotomy"]
        else:
            self.allow_autotomy = True 

        self.goal_counter = np.array([0])
        self.add_robot(body, connections)
        
        super().__init__(self.world)
        
        self.setup_action_space()


        self.default_viewer.track_objects("robot") 

    def get_obs(self):

        obs = np.concatenate([\
                self.get_vel_com_obs("robot"),\
                self.get_relative_pos_obs("robot"),\
                self.mode, self.goal_counter])

        return obs

    def add_robot(self, body, connections):

        self.robot_body = 1.0 * body
        self.robot_body_elements = self.robot_body.shape[0] * self.robot_body.shape[1]

        if self.mode:
            self.world.add_from_array("robot", body, 28, 1, connections=connections) 
        else:
            self.world.add_from_array("robot", body, 28, 1, connections=connections) 

    def setup_action_space(self):

        
        self.num_actuators = self.get_actuator_indices("robot").size 

        obs_size = self.get_obs().size

        action_space_size = self.num_actuators + self.robot_body_elements

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(action_space_size, ), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape = (obs_size,), dtype=float)

    def remove_robot(self, name="robot"):
        
        self.world.remove_object(name)

    def step(self, action):

        obs, reward, done, info  = 0, 0, True, {}

        position_1 = self.object_pos_at_time(self.get_time(), "robot")

        body_action = action[-self.robot_body_elements:]
        actuator_action = action[:self.num_actuators]

        done= super().step({"robot": actuator_action})

        position_2 = self.object_pos_at_time(self.get_time(), "robot")

        center_of_mass_1 = np.mean(position_1,1)
        center_of_mass_2 = np.mean(position_2,1)

        info = {"center_of_mass_1": center_of_mass_1,\
                "center_of_mass_2": center_of_mass_2,\
                "end_0": 0,\
                "end_1": 0,
                "autotomy_used": self.autotomy_used}

        reward = center_of_mass_2[0] - center_of_mass_1[0]

        reward *= -1.0 if self.mode else 1.0

        if done:
            print("***UNSTABLE SIMULATION***")
            print("   terminating with penalty -3 ")

            reward -= 3

        elif not(self.mode) and center_of_mass_2[0] >= self.goal[0]:
            reward += 1

            if self.goal_counter >= 1:
                self.reverse_direction(action)
            else: 
                self.goal_counter += 1

            info["end_0"] = 0

        elif self.mode and center_of_mass_2[0] <= self.goal[1]:
            
            reward += 1
            time_bonus = self.max_episode_steps - self.get_time()
            reward += 10 * time_bonus/self.max_episode_steps

            done = True

            info["end_1"] = 1

        obs = self.get_obs()
        
        return obs, reward, done, info

    def filter_robot_body(self, autotomy):
        # remove non-contiguous active pixels

        old_body = copy.deepcopy(self.robot_body)

        self.robot_body *= autotomy
        mask = label(self.robot_body)

        most = 0
        keep_index = 100
        for check in range(1,np.max(mask[0])):

            temp = (mask[0] == check).sum()

            if temp > most:
                most = temp
                keep_index = check

        if ((self.robot_body > 0) + (1 - (mask[0] == keep_index))).max() >= 2:
            self.autotomy_used = True

        if (0 < self.robot_body * (mask[0] == keep_index)).sum() >= 3:

            self.robot_body *= (mask[0] == keep_index) 

        if self.robot_body.sum() == 0:
            # no empty bodies
            self.robot_body = old_body
            self.autotomy_used = False
        elif self.robot_body.max() <= 2:
            # no passive robots
            self.robot_body = old_body
            self.autotomy_used = False
        if not check_connected(self.robot_body):
            # no disconnected body plans
            self.robot_body = old_body
            self.autotomy_used = False


        self.robot_body = 1.0 * np.clip(self.robot_body, 0, 4)
        
        self.add_robot(self.robot_body, connections=None)

    def reverse_direction(self, action):

        self.close()

        body_action = action[-self.robot_body_elements:]
        autotomy = 1.0 * (body_action > 0.5).reshape(self.robot_body.shape)

        this_filepath = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

        if self.use_difficulty and self.difficulty_level == 0: 
            filepath = os.path.join(this_filepath,  "world_data", \
                    "flat_walk_diff1.json")
        else:
            filepath = os.path.join(this_filepath,  "world_data", "flat_walk.json")

        self.world = EvoWorld.from_json(filepath)

        if self.allow_autotomy:
            self.filter_robot_body(autotomy)
        else:
            self.filter_robot_body(np.ones(self.robot_body.shape))

        self.goal_counter = np.array([0])

        super(BackAndForthEnvClass, self).__init__(self.world)
            
        if "robot" not in self.world.objects.keys():
            assert False, "no robot found"

        self.setup_action_space()
        self.default_viewer.track_objects("robot") 

        self.reset()

        self.mode = np.array([1])
        self.goal_counter = np.array([0])


    def reset(self):

        super().reset()

        self.mode = np.array([0])
        self.goal_counter = np.array([0])

        obs = self.get_obs()

        self.autotomy_used = False
        
        return obs

