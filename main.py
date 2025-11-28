import random
import torch
import numpy as np
from MARL_QAS_MaxCut import QAS
from MARL_QAS_Schwinger import VQE_QAS
import argparse
from replay_buffer import ReplayBuffer
from agent_net import Agent_net
from normalization import Normalization
from aim import Run

import sys
import pickle


class Runner_MaxCut:
    def __init__(self, args, seed, num_qubits, num_agents, max_ep_len, num_train_graphs,num_test_graphs, save_circ,
                 independent, verbose=True):
        self.args = args
        self.seed = seed
        self.save_circ = save_circ
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        # Create env
        self.num_agents = num_agents
        self.independent = independent
        self.dependency_string = 'separate' if self.independent else 'grouped'
        self.env = QAS(num_qubits=num_qubits, num_agents=self.num_agents,
                       max_episode_len=max_ep_len, independent_angles=self.independent,
                       num_graphs=num_train_graphs,
                       num_test_graphs=num_test_graphs)
        self.env.reset()
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        if verbose:
            print("number of agents={}".format(self.args.N))
            print("obs_dim={}".format(self.args.obs_dim))
            print("state_dim={}".format(self.args.state_dim))
            print("action_dim={}".format(self.args.action_dim))
            print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = Agent_net(self.args)

        self.replay_buffer = ReplayBuffer(self.args)

        self.aim_run = Run(experiment=f'MaxCut_{self.dependency_string}_{num_qubits}q_{num_agents}a_{max_ep_len}l')
        self.agent_n.aim_run = self.aim_run

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.total_steps = 0
        if self.args.use_reward_norm:
            if verbose:
                print("------using reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode(evaluate=False)  # Run an episode, store transitions
            self.total_steps += episode_steps
            self.aim_run.track(self.total_steps, name='total_steps')
            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        print(
            f'Total steps: {self.total_steps}, Eval reward: {evaluate_reward}, found solution {self.env.ar_found} times')  # win_times for inference

    def run_episode(self, evaluate=False):
        reach_tag = False
        episode_reward = 0
        self.env.reset()
        if self.args.use_rnn: # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            a_n = np.array(a_n)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            obs_n, r, _, done, info = self.env.step(a_n)

            if done:
                ar = self.env.ar if self.args.schwinger else self.env.train_ar
                # print(f'AR: {ar:.2f}, r:{r:.2f}')
                self.aim_run.track(ar, name='A.R.')
                # print(f'ACTION HISTORY: {self.env.action_history}')

                if ar > self.env.ar_threshold:
                    if (not args.schwinger and self.env.test_ar>0.95) or args.schwinger:
                            reach_tag = True
                            print('/'*40)
                            print(f'AR: {ar:.2f}, r:{r:.2f}')
                            print(f'TEST AR: {None if args.schwinger else self.env.test_ar }')
                            print(f'Step: {self.total_steps}')
                            # if self.save_circ:
                            #     with open(f'circuit_{self.env.num_qubits}q_{ar:.2f}_{self.num_agents}a_{self.env.max_episode_len}l_'
                            #               f'{self.dependency_string}_{self.env.total_num_episodes}step.pickle', 'wb') as file:
                            #         pickle.dump(self.env.total_qc, file)
                    reach_tag = True

            episode_reward += r/self.env.max_episode_len

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

        self.aim_run.track(episode_reward, name='episode_reward')

        return reach_tag, episode_reward, episode_step + 1


class Runner_Schwinger(Runner_MaxCut):
    def __init__(self, args, seed, num_qubits, num_agents, max_ep_len, m, save_circ,
                 independent):
        super(Runner_Schwinger, self).__init__(args=args,
                                               seed=seed,
                                               max_ep_len=max_ep_len,
                                               save_circ=save_circ,
                                               independent=independent,
                                               num_qubits=num_qubits,
                                               num_agents=num_agents,
                                               num_train_graphs=0,
                                               num_test_graphs=0,
                                               verbose=False)
        self.env = VQE_QAS(
            num_qubits=num_qubits, num_agents=self.num_agents,
            max_episode_len=max_ep_len, independent_angles=self.independent,
            m=m,
        )
        self.env.reset()
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        self.aim_run = Run(experiment=f'Schwinger_m{m}_{self.dependency_string}_{num_qubits}q_{num_agents}a_{max_ep_len}l')
        self.agent_n.aim_run = self.aim_run





if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=2, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e4), help="Model save frequency")
    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=600,
                        help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32,
                        help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64,
                        help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True,
                        help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=150, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    parser.add_argument("--num_qubits", type=int, default=4, required=True)
    parser.add_argument("--num_agents", type=int, default=1, required=True)
    parser.add_argument("--max_ep_len", type=int, default=5)
    parser.add_argument("--num_train_graphs", type=int, default=10)
    parser.add_argument("--num_test_graphs", type=int, default=10)
    parser.add_argument("--m", type=float, default=1)
    # parser.add_argument("--graph_type", type=str,choices=['3reg','erdos'], default='3reg')
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--save_circuit", action='store_true')
    parser.add_argument("--independent", action='store_true')
    parser.add_argument("--schwinger", action='store_true')
    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    if args.schwinger:
        runner = Runner_Schwinger(
            args,
            seed=args.seed,
            num_qubits=args.num_qubits,
            num_agents=args.num_agents,
            m=args.m,
            max_ep_len=args.max_ep_len,
            save_circ=args.save_circuit,
            independent=args.independent
        )
    else:
        runner = Runner_MaxCut(args,
                               seed=args.seed,  #999
                               num_qubits=args.num_qubits,
                               num_agents=args.num_agents,
                               num_train_graphs=args.num_train_graphs,
                               num_test_graphs=args.num_test_graphs,
                               max_ep_len=args.max_ep_len,
                               save_circ=args.save_circuit,
                               independent=args.independent,
                               )

    runner.run()
