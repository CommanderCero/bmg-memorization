from .ptb_env import PressTheButtonEnv

import gym

if 'PressTheButton-v0' not in gym.envs.registration.registry.env_specs:
    gym.envs.register(
        id='PressTheButton-v0',
        entry_point='gym_ptb:PressTheButtonEnv',
        kwargs={
            'gridworld': 'easy'
        }
    )

__all__ = [
    PressTheButtonEnv
]