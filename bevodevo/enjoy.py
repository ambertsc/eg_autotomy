import os
import sys
import argparse
import subprocess

import torch
import numpy as np
import time

import gym
import pybullet
import pybullet_envs

import matplotlib.pyplot as plt
import skimage


from mpi4py import MPI
comm = MPI.COMM_WORLD

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.mlps import MLPPolicy, \
        HebbianMLP, ABCHebbianMLP 
from bevodevo.policies.body_mlps import MLPBodyPolicy,\
        HebbianMLPBody, ABCHebbianMLPBody
from bevodevo.policies.body_mlps2 import MLPBodyPolicy2


from bevodevo.algos.es import ESPopulation
from bevodevo.algos.cmaes import CMAESPopulation
from bevodevo.algos.pges import PGESPopulation
from bevodevo.algos.ga import GeneticPopulation
from bevodevo.algos.random_search import RandomSearch

import eg_envs
from eg_auto.helpers import make_gif

def enjoy(argv):

    if "elite_pop" not in argv.file_path and ".pt" not in argv.file_path:
        my_dir = os.listdir(argv.file_path)
        latest_gen = 0
        my_file_path = ""
        for filename in my_dir:
            if "elite_pop" in filename:
                current_gen = int(filename.split("_")[5]) 
                if latest_gen < current_gen: 
                    latest_gen = current_gen
                    my_file_path = os.path.join(argv.file_path, filename)
    else:
        my_file_path = argv.file_path

    print(my_file_path)

    if "gatedrnn" in argv.policy.lower():
        policy_fn = GatedRNNPolicy
        argv.policy = "GatedRNNPolicy" 
    elif "impala" in argv.policy.lower():
        policy_fn = ImpalaCNNPolicy
        argv.policy = "ImpalaCNNPolicy"
    elif "cppnmlp" in argv.policy.lower():
        policy_fn = CPPNMLPPolicy
        arg.policy = "CPPNMLPPolicy"
    elif "abchebbianmlpbody" in argv.policy.lower():
        policy_fn = ABCHebbianMLPBody
        argv.policy = "ABCHebbianMLPBody"
    elif "abchebbianmlp" in argv.policy.lower():
        policy_fn = ABCHebbianMLP
        argv.policy = "ABCHebbianMLP"
    elif "cppnhebbianmlp" in argv.policy.lower():
        policy_fn = CPPNHebbianMLP
        argv.policy = "CPPNHebbianMLP"
    elif "hebbiancamlp2" in argv.policy.lower():
        policy_fn = HebbianCAMLP2
    elif "hebbiancamlp" in argv.policy.lower():
        policy_fn = HebbianCAMLP
    elif "hebbianmlpbody" in argv.policy.lower():
        policy_fn = HebbianMLPBody
        argv.policy = "HebbianMLPBody"
    elif "hebbianmlp" in argv.policy.lower():
        policy_fn = HebbianMLP
        argv.policy = "HebbianMLP"
    elif "mlpbodypolicy2" in argv.policy.lower():
        policy_fn = MLPBodyPolicy2
        argv.policy = "MLPBodyPolicy2"
    elif "mlpbodypolicy" in argv.policy.lower():
        policy_fn = MLPBodyPolicy
        argv.policy = "MLPBodyPolicy"
    elif "mlppolicy" in argv.policy.lower():
        policy_fn = MLPPolicy
        argv.policy = "MLPPolicy"
    else:
        assert False, "policy not found, check spelling?"


    if ".npy" in my_file_path:
        my_data = np.load(my_file_path, allow_pickle=True)[np.newaxis][0]
        env_name = my_data["env_name"]
    else:
        env_name = argv.env_name

    env = gym.make(env_name)

    if argv.no_render:
        gym_render = False
    else:
        if "BulletEnv" in env_name:
            env.render()
            gym_render = False
        else:
            gym_render = True

    obs_dim = env.observation_space.shape

    if len(obs_dim) == 3:
        obs_dim = obs_dim
    else:
        obs_dim = obs_dim[0]
    hid_dim = 16 #[32,32] 


    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    no_array = act_dim == 2 and discrete 

    if ".npy" in my_file_path:
        parameters = np.load(my_file_path, allow_pickle=True)[np.newaxis][0]
    else:
        parameters = torch.load(my_file_path)

    if type(parameters) is dict:
        elite_keep = len(parameters)
    else:
        elite_keep = 1

    for agent_idx in range(argv.agent_idx, argv.agent_idx \
            + min([argv.num_agents, elite_keep-argv.agent_idx])):

        if type(parameters) is dict:
            kwargs = dict(argv._get_kwargs())
            agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, \
                    "dim_y": act_dim, "params": parameters["elite_0"]} 
            if "body_dim" in kwargs.keys(): 
                agent_args["body_dim"] = argv.body_dim
            my_params = agent_args["params"]

            if "mode" in kwargs.keys():
                agent_args["mode"] = kwargs["mode"]

            agent_args["params"] = None
        else:
            agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, \
                    "dim_y": act_dim, "params": parameters} 
            if "body_dim" in dict(argv._get_kwargs()).keys():
                agent_args["body_dim"] = argv.body_dim
            my_params = agent_args["params"]
            agent_args["params"] = None
            if ".pt" in my_file_path:
                agent_args["params"] = None
                kwargs = dict(argv._get_kwargs())
                if "body_dim" in kwargs.keys(): 
                    agent_args["body_dim"] = argv.body_dim
                if "mode" in kwargs.keys(): 
                    agent_args["mode"] = kwargs["mode"]




        agent_args["discrete"] = discrete
        agent_args["body_dim"] = argv.body_dim
        agent = policy_fn(**agent_args)


        if ".pt" in my_file_path:
            agent.load_state_dict(parameters)
        else:
            agent.set_params(my_params)


        epd_rewards = []
        for episode in range(argv.episodes):
            if "BackAndForthEnv" in env_name and "body" in dir(agent):

                body = agent.get_body()

                env = gym.make(id=env_name, body=body, \
                        allow_autotomy=argv.use_autotomy, **kwargs) 

                env.seed(13)
                env.unwrapped.seed(13)

            else:
                env = gym.make(id=env_name)
            obs = env.reset()
            sum_reward = 0.0
            done = False
            step_count = 0
            while not done:

                action = agent.get_action(obs)


                if no_array and type(action) == np.ndarray\
                        or len(action.shape) > 1:

                    action = action[0]

                obs, reward, done, info = env.step(action)
                    

                step_count += 1

                sum_reward += reward

                if gym_render:
                    env.render()
                    #time.sleep(1e-2)

                if argv.save_frames and (step_count % argv.save_frames == 0):
                    
                    if "BulletEnv" in argv.env_name:
                        env.unwrapped._render_width = 640
                        env.unwrapped._render_height = 480

                    if "BackAndForthEnv" in argv.env_name:
                        img = env.render(mode="img")
                    else:
                        img = env.render(mode="rgb_array")

                    image_path = f"./frames/frame_agent{agent_idx}_epd{episode}_step{str(step_count).zfill(4)}.png"

                    scale_factor = 0

                    if float(argv.save_gif) < 1.0 and float(argv.save_gif) != 0.0:
                        my_scale = np.clip(float(argv.save_gif), 0.1, 0.9)
                        scale_factor = int(1/my_scale)

                    if scale_factor:
                        img = skimage.transform.resize(img, \
                                [elem//scale_factor for elem in img.shape[:-1]],\
                                anti_aliasing=True)

                    
                    img = 255. * img / np.max(img) 

                    skimage.io.imsave(image_path, np.array(img, dtype=np.uint8))

                time.sleep(0.01)
                if step_count >= argv.max_steps:
                    done = True
            print(f"env steps: {step_count} of {env.max_episode_steps}")
            print(f"autotomy used in env? {env.unwrapped.autotomy_used}")
            print(f"autotomy allowed in env? {env.unwrapped.allow_autotomy}")
            print(f"episode solved? {np.mean(sum_reward) > 32}")
            print(agent.body_dim, agent.__class__)


            epd_rewards.append(sum_reward)

        print("reward stats for elite {} over {} epds:".format(agent_idx, argv.episodes))
        print("mean rew: {:.3e}, +/- {:.3e} std. dev.".format(np.mean(epd_rewards), np.std(epd_rewards)))
        print("max rew: {:.3e}, min rew: {:.3e}".format(np.max(epd_rewards), np.min(epd_rewards)))

        if argv.save_gif:
            speedup = 3 if argv.save_frames==1 else 1
            speedup = [speedup, argv.save_frames]
            gif_tag = f"u{env.allow_autotomy}_m{agent.mode}_{os.path.split(argv.file_path)[-1]}"

            make_gif(tag=gif_tag, speedup=speedup)

            

        env.close()
    
    return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment parameters")

    parser.add_argument("-a", "--num_agents", type=int,\
            help="how many agents to evaluate", \
            default=1)
    parser.add_argument("-b", "--body_dim", type=int,\
            help="body dim", \
            default=8)
    parser.add_argument("-d", "--use_difficulty", type=int, default=0,\
            help="use increased difficulty")
    parser.add_argument("-e", "--episodes", type=int,\
            help="number of episodes", default=5)
    parser.add_argument("-f", "--file_path", type=str,\
            help="file path to model parameters", \
            default="./results/test_exp/")
    parser.add_argument("-g", "--save_gif", type=float, default=0,\
            help="1 - save gif to ./assets, 0 - do not.") 
    parser.add_argument("-i", "--agent_idx", type=int,\
            help="agent index to start with", default=0)
    parser.add_argument("-m", "--mode", default=0,\
            help="mode (0,1,2, or 3) for body co-evolution")
    parser.add_argument("-ms", "--max_steps", type=int,\
            help="maximum number of steps per episode", default=4000)
    parser.add_argument("-n", "--env_name", type=str, \
            help="name of environemt", default="BackAndForthEnv-v0")
    parser.add_argument("-nr", "--no_render", type=bool,\
            help="don't render", default=False)
    parser.add_argument("-pi", "--policy", type=str,\
            help="name of policy architecture", default="MLPPolicy")
    parser.add_argument("-s", "--save_frames", type=int, \
            help="save frames or not", default=0)
    parser.add_argument("-u", "--use_autotomy", type=int, default=1,\
            help="allow autotomy in training (for envs that support it)")



    args = parser.parse_args()

    enjoy(args)
