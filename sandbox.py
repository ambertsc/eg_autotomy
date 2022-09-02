import os
import copy
from copy import deepcopy

import numpy as np

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot

import gym
import envs


if __name__ == '__main__':
    np.random.seed(13)

    body, connections = sample_robot((4,4))
    env = gym.make("BackAndForthEnv-v0", body=body)
    #env = gym.make("BackAndForthEnv-v0", body=body, connections=connections)

    

    best_score = -float("inf")

    for trials in range(520):


        body, connections = sample_robot((6,6))

        env = gym.make("BackAndForthEnv-v0", body=deepcopy(body))

        _ = env.reset()
        total_reward = 0.
        pruning = np.random.randint(0,2, size=(env.robot_body_elements,))
        done= False

        while not done:
            if body.sum() == 0:
                print("boo")
                import pdb; pdb.set_trace()

            action = env.action_space.sample()

            action[-env.robot_body_elements:] = pruning

            obs, reward, done, info = env.step(action)
            total_reward += reward

        env.reset()

        if total_reward > best_score:

            best_body = 1 * body
            best_score = 1 * total_reward
            if best_score >= 12:
                best_pruning = 1 * pruning
            else:
                best_pruning = 0 * pruning

            
            best_connections = 1 * connections
            print(f"new best score: {best_score:.3f}")

        env.close()

            
    

    import pdb; pdb.set_trace()
    my_run = 1
    done = 0
    try:
        env = gym.make("BackAndForthEnv-v0", body=deepcopy(best_body))
    except:
        env = gym.make("BackAndForthEnv-v0", body=deepcopy(body))

    _ = env.reset()
    while my_run:
        while not done:
            action = env.action_space.sample()
            action[-env.robot_body_elements:] = best_pruning
            try:
                obs, reward, done, info = env.step(action)
            except:
                import pdb; pdb.set_trace()
            env.render()
            #if done:
            #    env.reset()
        env.reset()
        import pdb; pdb.set_trace()
    env.close()
