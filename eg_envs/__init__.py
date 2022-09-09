
# import envs and necessary gym packages
from gym.envs.registration import register
from eg_envs.back_and_forth_env import BackAndForthEnvClass
from eg_envs.adaptive_walk_env import AdaptiveWalkEnvClass

# register the env using gym's interface

register(
    id = 'AdaptiveWalkEnv-v0',
    entry_point = 'eg_envs.adaptive_walk_env:AdaptiveWalkEnvClass',
    max_episode_steps = 1024
)

register(
    id = 'BackAndForthEnv-v0',
    entry_point = 'eg_envs.back_and_forth_env:BackAndForthEnvClass',
    max_episode_steps = 2048
)
