from .ptb_env import PressTheButtonEnv

import gym

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