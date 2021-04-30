import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
from executor import Executor


from constants.ale import CONFIG_SETS as CONFIG_SETS_ALE

import run_utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # If --config-set is passed, it loads a predefined config and overwrites env-name, seed.
    # Otherwise, it does complete parsing (of all arguments).

    parser.add_argument('--config-set', type=int, default=None)
    parser.add_argument('--env-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--process-index', type=int, default=0)
    parser.add_argument('--machine-name', type=str, default='NA')
    parser.add_argument('--n-training-frames', type=int, default=10000000)
    parser.add_argument('--n-evaluation-trials', type=int, default=10)
    parser.add_argument('--evaluation-period', type=int, default=50000)
    parser.add_argument('--evaluation-visualization-period', type=int, default=200)

    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--save-models', action='store_true', default=False)
    parser.add_argument('--visualization-period', type=int, default=100)
    parser.add_argument('--model-save-period', type=int, default=200000)
    parser.add_argument('--env-training-seed', type=int, default=0)
    parser.add_argument('--env-evaluation-seed', type=int, default=1)

    parser.add_argument('--dqn-gamma', type=float, default=0.99)
    parser.add_argument('--dqn-rm-type', type=str, default='uniform')
    parser.add_argument('--dqn-per-ims', action='store_true', default=False)
    parser.add_argument('--dqn-per-alpha', type=float, default=0.4)
    parser.add_argument('--dqn-per-beta', type=float, default=0.6)
    parser.add_argument('--dqn-rm-init', type=int, default=50000)
    parser.add_argument('--dqn-rm-max', type=int, default=1000000)
    parser.add_argument('--dqn-target-update', type=int, default=7500)
    parser.add_argument('--dqn-batch-size', type=int, default=32)
    parser.add_argument('--dqn-learning-rate', type=float, default=0.0000625)
    parser.add_argument('--dqn-train-per-step', type=int, default=1)
    parser.add_argument('--dqn-train-period', type=int, default=4)
    parser.add_argument('--dqn-adam-eps', type=float, default=0.00015)
    parser.add_argument('--dqn-eps-start', type=float, default=1.0)
    parser.add_argument('--dqn-eps-final', type=float, default=0.01)
    parser.add_argument('--dqn-eps-steps', type=int, default=250000)
    parser.add_argument('--dqn-huber-loss-delta', type=float, default=1.0)
    parser.add_argument('--dqn-hidden-size', type=int, default=512)

    parser.add_argument('--bc-batch-size', type=int, default=32)
    parser.add_argument('--bc-learning-rate', type=float, default=0.0001)
    parser.add_argument('--bc-adam-eps', type=float, default=0.00015)
    parser.add_argument('--bc-dropout-rate', type=float, default=0.35)
    parser.add_argument('--bc-hidden-size', type=int, default=512)
    parser.add_argument('--bc-uc-ensembles', type=int, default=100)

    parser.add_argument('--load-teacher', action='store_true', default=False)

    # Collection
    # 'none', 'early', 'random', 'uncertainty_based'
    parser.add_argument('--advice-collection-method', type=str, default='none')
    parser.add_argument('--advice-collection-budget', type=int, default=0)
    parser.add_argument('--advice-collection-uncertainty-threshold', type=float, default=0)

    # Imitation
    # 'none', 'periodic'
    parser.add_argument('--advice-imitation-method', type=str, default='none')
    parser.add_argument('--advice-imitation-period-steps', type=int, default=0)
    parser.add_argument('--advice-imitation-period-samples', type=int, default=0)
    parser.add_argument('--advice-imitation-training-iterations-init', type=int, default=0)
    parser.add_argument('--advice-imitation-training-iterations-periodic', type=int, default=0)

    parser.add_argument('--autoset-advice-uncertainty-threshold', action='store_true', default=False)

    # Reuse
    # 'none', 'restricted', 'extended'
    parser.add_argument('--advice-reuse-method', type=str, default='none')
    parser.add_argument('--advice-reuse-probability', type=float, default=0)
    parser.add_argument('--advice-reuse-uncertainty-threshold', type=float, default=0)
    parser.add_argument('--advice-reuse-probability-decay', action='store_true', default=False)
    parser.add_argument('--advice-reuse-probability-decay-begin', type=int, default=0)
    parser.add_argument('--advice-reuse-probability-decay-end', type=int, default=0)
    parser.add_argument('--advice-reuse-probability-final', type=float, default=0)

    # ------------------------------------------------------------------------------------------------------------------

    arg_vars = vars(parser.parse_args())

    if arg_vars['config_set'] is not None:
        print('Loading predefined config set #', arg_vars['config_set'], '...')
        print('Parsing ALE config...')
        config = run_utils.configToExecutorConfig(CONFIG_SETS_ALE[arg_vars['config_set']])
        config['env_name'] = arg_vars['env_name']
        config['seed'] = arg_vars['seed']
    else:
        print('Parsing full config...')
        config = vars(parser.parse_args())

    if config['run_id'] is None:
        config['run_id'] = run_utils.generateRunId(config['env_name'], arg_vars['config_set'])

    print(config['run_id'])
    run_utils.printConfig(config)

    env, eval_env = run_utils.generateEnvs(config['env_name'],
                                            config['env_training_seed'],
                                            config['env_evaluation_seed'])

    executor = Executor(config, env, eval_env)
    executor.run()
