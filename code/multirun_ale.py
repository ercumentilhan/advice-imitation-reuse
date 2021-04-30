import multiprocessing
import subprocess
import shlex
import time
import math
import socket

from constants.ale import CONFIG_SETS as CONFIG_SETS_ALE

import run_utils

lock = multiprocessing.Lock()

def work(cmd):
    lock.acquire()
    time.sleep(1)
    lock.release()
    return subprocess.call(shlex.split(cmd), shell=False)  # return subprocess.call(cmd, shell=False)


if __name__ == '__main__':

    machine_name = 'UNDEFINED'
    machine_id = None
    n_processors = 6

    hostname = socket.gethostname()

    if hostname == 'DESKTOP-LA8NF7N':  # HOME
        print('DESKTOP-LA8NF7 (HOME)')
        machine_name = 'HOME'
        n_processors = 9
        machine_id = 0
        config_set = 0
        n_seeds = 3

    elif hostname == 'DESKTOP-8A3QAR8':  # LAB-2
        print('DESKTOP-8A3QAR8 (LAB-2)')
        machine_name = 'LAB-2'
        n_processors = 4
        machine_id = 2
        config_set = 2
        n_seeds = 2

    elif hostname == 'DESKTOP-A90S73P':  # LAB-1
        print('DESKTOP-A90S73P (LAB-1)')
        machine_name = 'LAB-1'
        n_processors = 3
        machine_id = 1
        config_set = 1
        n_seeds = 4

    elif hostname == 'DESKTOP-DM48GMR':  # MSI Notebook
        print('DESKTOP-DM48GMR (MSI Notebook)')
        machine_name = 'MSI-NB'
        n_processors = 1
        machine_id = 3
        config_set = 3

    seeds_all = list(range((machine_id + 1) * 100 + 1, (machine_id + 1) * 100 + 21))

    # ------------------------------------------------------------------------------------------------------------------
    n_seeds = 1
    n_processors = 5

    # Envs to be experimented with:
    #env_names = ['ALE-Breakout', 'ALE-Enduro', 'ALE-Freeway', 'ALE-MsPacman', 'ALE-Pong', 'ALE-Qbert',
    #             'ALE-Seaquest', 'ALE-SpaceInvaders', 'ALE-Zaxxon']

    env_names = ['ALE-Seaquest'] #, 'ALE-Freeway', 'ALE-Pong', 'ALE-Qbert', 'ALE-Seaquest'

    #  (10k, 25k)

    # 1000
    # 2045
    # 3545 -> 6545
    # 5045 -> 7045

    # Freewat: 0.00009
    # Pong: 0.025740800425410257


    # 1000: Baseline training
    # 1900: Evaluate expert
    #
    # 2035, 2045: Early
    # 2135, 2145: Random
    #
    # 3035, 3045: AAMAS 0.0001
    # 3135, 3145: AAMAS 0.001
    # 3235, 3245: AAMAS 0.01
    #
    # 3535, 3545: *AAMAS Auto*
    # 5035, 5045: Final

    #
    # 7045

    run_config_idx = [6000]

    # ------------------------------------------------------------------------------------------------------------------

    i_parameter_set = 0
    i_command = 0
    commands = []

    #seeds = seeds_all[:n_seeds]
    seeds = [302]

    print('Seeds: {}\n'.format(seeds))

    for env_name in env_names:
        for run_config_id in run_config_idx:
            run_config = CONFIG_SETS_ALE[run_config_id]
            run_config[0]['env-name'] = env_name

            run_id = run_utils.generateRunId(run_config[0]['env-name'], run_config_id)

            run_config[0]['machine-name'] = str(machine_name)
            run_config[0]['process-index'] = str(i_command % n_processors)
            run_config[0]['run-id'] = str(run_id)

            for seed in seeds:
                seed_run_config = run_config.copy()
                seed_run_config[0]['seed'] = str(seed)
                commands.append(run_utils.configToCommand(seed_run_config))
                i_command += 1
            i_parameter_set += 1

    # ------------------------------------------------------------------------------------------------------------------

    # print(commands)

    print('There are {} commands.'.format(len(commands)))

    n_cycles = int(math.ceil(len(commands) / n_processors))

    print('There are {} cycles.'.format(n_cycles))

    for i_cycle in range(n_cycles):
        pool = multiprocessing.Pool(processes=n_processors)

        start = (n_processors * i_cycle)
        end = start + n_processors

        print('start and end:', start, end)

        if end > len(commands):
            end = len(commands)

        print('start and end:', start, end)

        print(pool.map(work, commands[(n_processors * i_cycle):(n_processors * i_cycle) + n_processors]))
