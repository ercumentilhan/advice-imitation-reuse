import os
import random
import pathlib
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from dqn_egreedy import EpsilonGreedyDQN
from run_statistics import Statistics

os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'

from constants.general import *
import gym_video_recorder

from behavioural_cloning import BehaviouralCloning


class Executor:
    def __init__(self, config, env, eval_env) -> None:
        self.config = config
        self.env = env
        self.eval_env = eval_env

        self.stats = None

        self.student_agent = None
        self.teacher_agent = None

        self.save_videos_path = None

        self.steps_reward = 0.0
        self.steps_reward_real = 0.0
        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_reward_real = 0.0

        self.run_id = None

        self.video_recorder = None

        self.session = None
        self.summary_writer = None
        self.saver = None

        # Action advising
        self.action_advising_budget = self.config['advice_collection_budget']
        self.reuse_enabled = False
        self.bc_model = None
        self.initial_imitation_is_performed = False
        self.steps_since_imitation = 0
        self.samples_since_imitation = 0

        self.advices_reused_ep = 0
        self.advices_reused_ep_correct = 0

        self.advice_reuse_probability_decrement = 0
        self.advice_reuse_probability_decay_steps = \
            self.config['advice_reuse_probability_decay_end'] - self.config['advice_reuse_probability_decay_begin']
        if self.config['advice_reuse_probability_decay']:
            self.advice_reuse_probability_decrement = \
                (self.config['advice_reuse_probability'] - self.config['advice_reuse_probability_final']) /\
                self.advice_reuse_probability_decay_steps

        self.advice_reuse_probability = self.config['advice_reuse_probability']

    # ------------------------------------------------------------------------------------------------------------------

    def run(self):
        os.environ['PYTHONHASHSEED'] = str(self.config['seed'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        tf.compat.v1.set_random_seed(self.config['seed'])
        tf.random.set_seed(self.config['seed'])

        self.run_id = self.config['run_id']
        self.seed_id = str(self.config['seed'])

        print('Run ID: {}'.format(self.run_id))

        # --------------------------------------------------------------------------------------------------------------

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.join(str(pathlib.Path(scripts_dir).parent))

        print('{} (Code directory)'.format(scripts_dir))
        print('{} (Workspace directory)'.format(workspace_dir))

        summaries_dir = os.path.join(workspace_dir, 'summaries')
        os.makedirs(summaries_dir, exist_ok=True)

        checkpoints_dir = os.path.join(workspace_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        videos_dir = os.path.join(workspace_dir, 'videos')
        os.makedirs(videos_dir, exist_ok=True)

        save_summary_path = os.path.join(summaries_dir, self.run_id, self.seed_id)
        save_model_path = os.path.join(checkpoints_dir, self.run_id, self.seed_id)
        self.save_videos_path = os.path.join(videos_dir, self.run_id, self.seed_id)

        if self.config['save_models']:
            os.makedirs(save_model_path, exist_ok=True)

        os.makedirs(self.save_videos_path, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['use_gpu']:
            print('Using GPU.')
            session_config = tf.compat.v1.ConfigProto(
                #intra_op_parallelism_threads=1,
                #inter_op_parallelism_threads=1
                )
            session_config.gpu_options.allow_growth = True
        else:
            print('Using CPU.')
            session_config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                device_count={'CPU': 1, 'GPU': 0})

        self.session = tf.compat.v1.InteractiveSession(graph=tf.compat.v1.get_default_graph(), config=session_config)

        self.summary_writer = tf.compat.v1.summary.FileWriter(save_summary_path, self.session.graph)

        self.stats = Statistics(self.summary_writer, self.session)
        self.teacher_stats = Statistics(self.summary_writer, self.session)

        # --------------------------------------------------------------------------------------------------------------

        self.env_info = {}
        env_info = ENV_INFO[self.config['env_name']]
        self.env_info['max_timesteps'] = env_info[8]

        self.config['env_type'] = env_info[1]
        self.config['env_obs_form'] = env_info[2]
        self.config['env_states_are_countable'] = env_info[3]

        self.config['env_obs_dims'] = self.env.observation_space.shape
        self.config['env_n_actions'] = self.env.action_space.n
        self.config['env_obs_dims'] = (84, 84, 4)  # LazyFrames is enabled

        print(self.config['env_name'])
        print(self.config['env_obs_dims'], self.config['env_n_actions'], '\n')

        self.config['rm_extra_content'] = ['source', 'expert_action', 'preserve']

        # --------------------------------------------------------------------------------------------------------------
        # Setup student agent
        self.config['student_id'] = self.run_id

        self.student_agent = EpsilonGreedyDQN(self.config['student_id'], self.config, self.session,
                                              self.config['dqn_eps_start'],
                                              self.config['dqn_eps_final'],
                                              self.config['dqn_eps_steps'], self.stats)

        self.config['student_id'] = self.student_agent.id

        print('Student ID: {}'.format(self.student_agent.id))

        # --------------------------------------------------------------------------------------------------------------
        # Setup teacher agent

        if self.config['load_teacher']:
            teacher_info = TEACHER[self.config['env_name']]
            self.config['teacher_id'] = teacher_info[0]
            self.teacher_agent = EpsilonGreedyDQN(self.config['teacher_id'],
                                                       self.config,
                                                       self.session,
                                                       eps_start=0.0, eps_final=0.0, eps_steps=1,
                                                       stats=self.stats)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['advice_imitation_method'] != 'none':
            print('Initialising behaviour cloning network...')
            self.bc_model = BehaviouralCloning('BHC', self.config, self.session, None)

        # --------------------------------------------------------------------------------------------------------------

        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Number of parameters: {}'.format(total_parameters))

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
        self.session.run(tf.compat.v1.global_variables_initializer())

        # --------------------------------------------------------------------------------------------------------------
        # Restore the teacher policy
        if self.config['load_teacher']:
            print('Restoring teacher...')
            teacher_info = TEACHER[self.config['env_name']]
            self.teacher_agent.restore(checkpoints_dir, teacher_info[0] + '/' + teacher_info[1], teacher_info[2])
            print('Done')

        # --------------------------------------------------------------------------------------------------------------

        print('Finalising')
        if not self.config['save_models']:
            tf.compat.v1.get_default_graph().finalize()

        eval_score, eval_score_real = self.evaluate()
        print('Evaluation @ {} | {} & {}'.format(self.stats.n_env_steps, eval_score, eval_score_real))

        obs, render = self.reset_env()

        while True:
            action = None

            self_action, action_is_explorative = self.student_agent.get_action(obs)

            if action_is_explorative:
                self.stats.exploration_steps_taken += 1
                self.stats.exploration_steps_taken_episode += 1
                self.stats.exploration_steps_taken_cum += 1

            # ----------------------------------------------------------------------------------------------------------

            advice_collection_occurred = False

            if self.config['advice_collection_method'] != 'none' and self.action_advising_budget > 0:

                if self.config['advice_collection_method'] == 'early':
                    advice_collection_occurred = True

                elif self.config['advice_collection_method'] == 'random':
                    if random.random() < 0.5:
                        advice_collection_occurred = True
                    #self.selected_policy = 1

                elif self.config['advice_collection_method'] == 'uncertainty_based':
                    if self.initial_imitation_is_performed:
                        bc_uncertainty = self.bc_model.get_uncertainty(obs)
                        if bc_uncertainty > self.config['advice_collection_uncertainty_threshold']:
                            advice_collection_occurred = True
                    else:
                        advice_collection_occurred = True

            if advice_collection_occurred:
                self.action_advising_budget -= 1
                self.stats.advices_taken += 1
                self.stats.advices_taken_cum += 1

                action = self.teacher_agent.get_greedy_action(obs)

                if self.config['advice_imitation_method'] != 'none':
                    self.bc_model.feedback_observe(obs, action)
                    self.samples_since_imitation += 1

            self.steps_since_imitation += 1

            # ----------------------------------------------------------------------------------------------------------

            # Imitation
            if self.config['advice_imitation_method'] == 'periodic':

                if (self.steps_since_imitation >= self.config['advice_imitation_period_steps'] and
                    self.samples_since_imitation >= (self.config['advice_imitation_period_samples'] / 2)) or \
                        self.samples_since_imitation >= self.config['advice_imitation_period_samples']:

                    print(self.steps_since_imitation, self.samples_since_imitation)

                    if not self.initial_imitation_is_performed:
                        train_behavioural_cloner(self.bc_model,
                                                 self.config['advice_imitation_training_iterations_init'])

                        print('Self evaluating model...')
                        uc_threshold, accuracy = evaluate_behavioural_cloner(self.bc_model)

                        if self.config['autoset_advice_uncertainty_threshold']:
                            self.config['advice_reuse_uncertainty_threshold'] = uc_threshold
                            self.config['advice_collection_uncertainty_threshold'] = uc_threshold

                        self.initial_imitation_is_performed = True
                        self.steps_since_imitation = 0
                        self.samples_since_imitation = 0
                    else:
                        if self.bc_model.replay_memory.__len__() == self.config['advice_collection_budget']:
                            train_behavioural_cloner(self.bc_model,
                                                          self.config['advice_imitation_training_iterations_init'])
                        else:
                            train_behavioural_cloner(self.bc_model,
                                                          self.config['advice_imitation_training_iterations_periodic'])

                        print('Self evaluating model...')
                        uc_threshold, accuracy = evaluate_behavioural_cloner(self.bc_model)

                        if self.config['autoset_advice_uncertainty_threshold']:
                            print('setting uc threshold:', uc_threshold)
                            self.config['advice_reuse_uncertainty_threshold'] = uc_threshold
                            self.config['advice_collection_uncertainty_threshold'] = uc_threshold

                        self.steps_since_imitation = 0
                        self.samples_since_imitation = 0

            # ----------------------------------------------------------------------------------------------------------
            # Reuse
            reuse_advice = False
            if not advice_collection_occurred:
                if self.config['advice_reuse_method'] != 'none' and \
                        self.initial_imitation_is_performed:

                    if self.config['advice_reuse_method'] == 'extended' or \
                            (self.config['advice_reuse_method'] == 'restricted' and action_is_explorative):

                        if self.reuse_enabled:
                            bc_uncertainty = self.bc_model.get_uncertainty(obs)
                            if bc_uncertainty < self.config['advice_reuse_uncertainty_threshold']:
                                reuse_advice = True

            if reuse_advice:
                action = np.argmax(self.bc_model.get_action_probs(obs))
                self.stats.advices_reused += 1
                self.stats.advices_reused_cum += 1

                self.advices_reused_ep += 1
                self.stats.advices_reused_ep_cum += 1

                # To measure accuracy of imitation - can be disabled to speed-up execution
                if action == self.teacher_agent.get_greedy_action(obs):
                    self.stats.advices_reused_correct += 1
                    self.stats.advices_reused_correct_cum += 1
                    self.advices_reused_ep_correct += 1
                    self.stats.advices_reused_ep_correct_cum += 1

            # ----------------------------------------------------------------------------------------------------------
            if action is None:
                action = self_action

            if advice_collection_occurred:
                source = 1
            else:
                source = 0

            # ----------------------------------------------------------------------------------------------------------
            # Execute action
            obs_next, reward, done, info, _ = self.env.step(action)
            reward_real = reward

            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'obs_next': obs_next,
                'done': done,
                'source': source,
                'expert_action': None,
                'preserve': False
            }

            if render:
                self.video_recorder.capture_frame()

            self.episode_reward += reward
            self.episode_reward_real += reward_real
            self.episode_duration += 1

            self.steps_reward += reward
            self.steps_reward_real += reward_real
            self.stats.n_env_steps += 1

            # ----------------------------------------------------------------------------------------------------------
            # Feedback
            self.student_agent.feedback_observe(transition)

            # ----------------------------------------------------------------------------------------------------------

            td_error_batch, loss = self.student_agent.feedback_learn()

            if self.config['advice_reuse_probability_decay'] and \
                    self.stats.n_env_steps > self.config['advice_reuse_probability_decay_begin'] and \
                    self.advice_reuse_probability > self.config['advice_reuse_probability_final']:
                self.advice_reuse_probability -= self.advice_reuse_probability_decrement
                if self.advice_reuse_probability < self.config['advice_reuse_probability_final']:
                    self.advice_reuse_probability = self.config['advice_reuse_probability_final']

            self.stats.loss += loss

            obs = obs_next
            done = done or self.episode_duration >= self.env_info['max_timesteps']

            if done:
                self.stats.n_episodes += 1
                self.stats.episode_reward_auc += np.trapz([self.stats.episode_reward_last, self.episode_reward])
                self.stats.episode_reward_last = self.episode_reward

                self.stats.episode_reward_real_auc += np.trapz([self.stats.episode_reward_real_last, self.episode_reward_real])
                self.stats.episode_reward_real_last = self.episode_reward_real

                self.stats.reuse_enabled_in_ep_cum += (1 if self.reuse_enabled != 0 else 0)

                self.stats.update_summary_episode(self.episode_reward, self.stats.episode_reward_auc,
                                                  self.episode_duration,
                                                  self.advices_reused_ep, self.advices_reused_ep_correct,
                                                  1 if self.reuse_enabled else 0,
                                                  self.episode_reward_real, self.stats.episode_reward_real_auc,)

                print('{}'.format(self.stats.n_episodes), end=' | ')
                print('{:.1f}'.format(self.episode_reward), end=' | ')
                print('{:.1f}'.format(self.episode_reward_real), end=' | ')
                print('{}'.format(self.episode_duration), end=' | ')
                print('{}'.format(self.stats.n_env_steps))

                if render:
                    self.video_recorder.close()
                    self.video_recorder.enabled = False

                obs, render = self.reset_env()

            # Per N steps summary update
            if self.stats.n_env_steps % self.stats.n_steps_per_update == 0:
                self.stats.steps_reward_auc += np.trapz([self.stats.steps_reward_last, self.steps_reward])
                self.stats.steps_reward_last = self.steps_reward
                self.stats.epsilon = self.student_agent.eps

                self.stats.steps_reward_real_auc += np.trapz([self.stats.steps_reward_real_last, self.steps_reward_real])
                self.stats.steps_reward_real_last = self.steps_reward_real

                self.stats.update_summary_steps(self.steps_reward, self.stats.steps_reward_auc,
                                                self.steps_reward_real, self.stats.steps_reward_real_auc)

                self.stats.exploration_steps_taken = 0

                self.stats.advices_taken = 0
                self.stats.advices_used = 0
                self.stats.advices_reused = 0
                self.stats.advices_reused_correct = 0

                self.steps_reward = 0.0
                self.steps_reward_real = 0.0


            if self.stats.n_env_steps % self.config['evaluation_period'] == 0:
                evaluation_score = self.evaluate()
                print('Evaluation ({}): {}'.format(self.stats.n_episodes, evaluation_score))

            if self.config['save_models'] and \
                    (self.stats.n_env_steps % self.config['model_save_period'] == 0 or
                     self.stats.n_env_steps >= self.config['n_training_frames']):
                self.save_model(save_model_path)

            if self.stats.n_env_steps >= self.config['n_training_frames']:
                break

        print('Env steps: {}'.format(self.stats.n_env_steps))

        self.session.close()

    # ==================================================================================================================

    def reset_env(self):
        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_reward_real = 0.0

        self.stats.advices_reused_episode = 0
        self.stats.advices_reused_correct_episode = 0
        self.stats.exploration_steps_taken_episode = 0

        self.advices_reused_ep = 0
        self.advices_reused_ep_correct = 0

        render = self.stats.n_episodes % self.config['visualization_period'] == 0
        if render:
            self.video_recorder = gym_video_recorder.\
                VideoRecorder(self.env,
                              base_path=os.path.join(self.save_videos_path, '{}_{}'.format(
                                  str(self.stats.n_episodes), str(self.stats.n_env_steps))))

        obs = self.env.reset()

        if render:
            self.video_recorder.capture_frame()

        if self.config['advice_reuse_method'] == 'none':
            self.reuse_enabled = False
        else:
            if self.config['advice_reuse_method'] == 'restricted' or \
                    self.config['advice_reuse_method'] == 'extended':
                # Enable/disable advice reuse in the next episode
                if random.random() < self.advice_reuse_probability:
                    self.reuse_enabled = True
                else:
                    self.reuse_enabled = False
            else:
                # Default: no valid method
                self.reuse_enabled = False

        return obs, render

    # ==================================================================================================================

    def evaluate(self):
        eval_render = self.stats.n_evaluations % self.config['evaluation_visualization_period'] == 0

        eval_total_reward_real = 0.0
        eval_total_reward = 0.0
        eval_duration = 0

        self.eval_env.seed(self.config['env_evaluation_seed'])

        if eval_render:
            video_capture_eval = gym_video_recorder.\
            VideoRecorder(self.eval_env, base_path=
            os.path.join(self.save_videos_path,
                         'E_{}_{}'.format(str(self.stats.n_episodes), str(self.stats.n_env_steps))))

        for i_eval_trial in range(self.config['n_evaluation_trials']):

            eval_obs = self.eval_env.reset()

            eval_episode_reward_real = 0.0
            eval_episode_reward = 0.0
            eval_episode_duration = 0

            while True:
                if eval_render:
                    video_capture_eval.capture_frame()

                eval_action = self.student_agent.get_greedy_action(eval_obs)

                eval_obs_next, eval_reward, eval_done, eval_info, eval_real_reward \
                    = self.eval_env.step(eval_action)

                eval_episode_reward_real += eval_real_reward
                eval_episode_reward += eval_reward

                eval_duration += 1
                eval_episode_duration += 1
                eval_obs = eval_obs_next

                eval_done = eval_done or eval_episode_duration >= self.env_info['max_timesteps']

                if eval_done:
                    if eval_render:
                        video_capture_eval.capture_frame()
                        video_capture_eval.close()
                        video_capture_eval.enabled = False
                        eval_render = False

                    eval_total_reward += eval_episode_reward
                    eval_total_reward_real += eval_episode_reward_real
                    break

        eval_mean_reward = eval_total_reward / float(self.config['n_evaluation_trials'])
        eval_mean_reward_real = eval_total_reward_real / float(self.config['n_evaluation_trials'])

        self.stats.evaluation_reward_auc += np.trapz([self.stats.evaluation_reward_last, eval_mean_reward])
        self.stats.evaluation_reward_last = eval_mean_reward

        self.stats.evaluation_reward_real_auc += np.trapz([self.stats.evaluation_reward_real_last, eval_mean_reward_real])
        self.stats.evaluation_reward_real_last = eval_mean_reward_real

        self.stats.n_evaluations += 1

        self.stats.update_summary_evaluation(eval_mean_reward,
                                             eval_duration,
                                             self.stats.evaluation_reward_auc, 
                                             eval_mean_reward_real,
                                             self.stats.evaluation_reward_real_auc)

        return eval_mean_reward, eval_mean_reward_real

    # ==================================================================================================================

    def save_model(self, save_model_path):
        model_path = os.path.join(os.path.join(save_model_path), 'model-{}.ckpt').format(
            self.stats.n_env_steps)
        print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
        self.saver.save(self.session, model_path)

# ======================================================================================================================

def train_behavioural_cloner(bc_model, n_iters):

    if bc_model.replay_memory.__len__() == 0:
        print('\nBehavioural cloner has 0 samples. Skipping training.')
    else:
        print('\nTraining behavioural cloner with {} samples for {} steps...'.format(
            bc_model.replay_memory.__len__(), n_iters))
        for _ in range(n_iters):
            bc_model.feedback_learn()

# ======================================================================================================================

def evaluate_behavioural_cloner(bc_model):
    n_samples = bc_model.replay_memory.__len__()

    if n_samples == 0:
        return np.inf, 0.01

    uc_values_all, uc_values_correct, uc_values_incorrect = [], [], []
    n_correct, n_incorrect = 0, 0

    for i in range(n_samples):
        bc_obs = bc_model.replay_memory._storage[i][0]
        bc_act = bc_model.replay_memory._storage[i][1]
        uncertainty = bc_model.get_uncertainty(bc_obs)
        prediction = np.argmax(bc_model.get_action_probs(bc_obs))

        uc_values_all.append(uncertainty)
        if bc_act == prediction:
            uc_values_correct.append(uncertainty)
            n_correct += 1
        else:
            uc_values_incorrect.append(uncertainty)
            n_incorrect += 1

    print('Post imitation analysis is completed.')
    print('All samples:')
    print('> Max:', np.max(uc_values_all))
    print('> Min:', np.min(uc_values_all))
    print('> Avg:', np.mean(uc_values_all))
    print('> 80%:', np.percentile(uc_values_all, 80))
    print('> 90%:', np.percentile(uc_values_all, 90))
    print('')
    if len(uc_values_correct) > 0:
        print('Correct samples:')
        print('> Max:', np.max(uc_values_correct))
        print('> Min:', np.min(uc_values_correct))
        print('> Avg:', np.mean(uc_values_correct))
        print('> 80%:', np.percentile(uc_values_correct, 80))
        print('> 90%:', np.percentile(uc_values_correct, 90))
    else:
        print('No correct samples.')
    print('')

    if len(uc_values_incorrect) > 0:
        print('Incorrect samples:')
        print('> Max:', np.max(uc_values_incorrect))
        print('> Min:', np.min(uc_values_incorrect))
        print('> Avg:', np.mean(uc_values_incorrect))
        print('> 80%:', np.percentile(uc_values_incorrect, 80))
        print('> 90%:', np.percentile(uc_values_incorrect, 90))
    else:
        print('No incorrect samples.')

    accuracy = (n_correct / (n_correct + n_incorrect)) * 100.0
    print('\nAccuracy:', accuracy)
    print('')

    return np.percentile(uc_values_correct, 90), accuracy


