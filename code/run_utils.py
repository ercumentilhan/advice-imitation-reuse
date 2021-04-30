from time import localtime, strftime
import gym
import frame_stack
import atari_preprocessing
import random
from constants.general import *


def generateRunId(env_name, config_set):
    run_id = str(ENV_INFO[env_name][0]) + '_' \
             + ('000' if config_set is None else str(config_set).zfill(3)) + '_' \
             + str(random.randint(0, 999)).zfill(3) + '_' \
             + strftime("%Y%m%d-%H%M%S", localtime())
    return run_id

# ==================================================================================================================

def generateEnvs(env_name, env_training_seed, env_evaluation_seed):
    env_info = ENV_INFO[env_name]
    env_name = env_info[4]

    env = gym.make(env_name)
    env = atari_preprocessing.AtariPreprocessing(env)
    env = frame_stack.FrameStack(env, num_stack=4)
    env.seed(env_training_seed)

    eval_env = gym.make(env_name)
    eval_env = atari_preprocessing.AtariPreprocessing(eval_env)
    eval_env = frame_stack.FrameStack(eval_env, num_stack=4)
    eval_env.seed(env_evaluation_seed)

    return env, eval_env

# ==================================================================================================================

def configToCommand(config):
    command = 'python main.py'
    for key in config[0]:
        command += ' --' + key + ' ' + str(config[0][key])
    for key in config[1]:
        if config[1][key]:
            command += ' --' + key
    return command

# ==================================================================================================================

def configToExecutorConfig(config):
    exec_config = {}
    for key in config[0]:
        exec_config[key.replace('-', '_')] = config[0][key]
    for key in config[1]:
        exec_config[key.replace('-', '_')] = config[1][key]
    return exec_config

# ==================================================================================================================

def printConfig(config):
    for key in config:
        print('{}: {}'.format(key, config[key]))
