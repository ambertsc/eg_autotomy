
# import envs and necessary gym packages
from gym.envs.registration import register
from envs.back_and_forth_env import BackAndForthEnvClass

# register the env using gym's interface

register(
    id = 'BackAndForthEnv-v0',
    entry_point = 'envs.back_and_forth_env:BackAndForthEnvClass',
    max_episode_steps = 1024
)
