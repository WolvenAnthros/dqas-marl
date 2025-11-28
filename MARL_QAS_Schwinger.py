import numpy as np
import random
from scipy.linalg import eigh
from gymnasium import Env
from qulacs import QuantumState, ParametricQuantumCircuit, Observable, PauliOperator
from scipy.optimize import minimize
import pickle
from schwinger import SchwingerHamiltonian


class VQE_QAS(Env):
    def __init__(
            self, num_qubits, num_agents, max_episode_len,
            m=1,
            debug=False,
            ar_threshold=0.95,
            independent_angles=False,
    ):
        """
        Initialize environment class. Most of the variables that need to be re-set between runs are in the reset() method.
        :param num_qubits:
        :param num_agents:
        :param max_episode_len: Max number of steps, determines circuit depth and possible amount of variable parameters
        :param num_graphs: How many MaxCut instances to use in the train set
        :param num_test_graphs: Same as above but for test
        :param graph_type: 3-regular or erdos (Erdos-Renyi)
        :param debug: verbose most important steps
        :param ar_threshold: When to start validating the solution on test set
        """
        self.num_qubits = num_qubits
        self.num_agents = num_agents
        self.max_episode_len = max_episode_len
        self.debug = debug
        self.ar_threshold = ar_threshold
        self.ar_found = 0  # how many times found satisfactory AR
        self.num_qubits_per_agent = self.num_qubits // self.num_agents
        self.state = QuantumState(num_qubits)
        self.agent_states = [QuantumState(self.num_qubits_per_agent) for _ in range(self.num_agents)]
        self.schwinger = SchwingerHamiltonian(self.num_qubits)
        self.m = m
        self.hamiltonian = self.schwinger.make_hamiltonian(m=m)  # Schwinger hamiltonian matrix
        self.min_energy, self.max_energy = self.get_min_max_energy()
        _, _, self.ground_state, _ = self.get_min_max_energy(get_state=True)
        self.independent = independent_angles
        self.total_num_episodes = 0

    def reset(self,
              seed=42,
              options=None):
        """
        Initialize circuits, prepare action space,
        :param seed:
        :param options:
        :return:
        """
        self.total_num_episodes += 1  # count how many times we called reset()
        self.total_qc = ParametricQuantumCircuit(self.num_qubits)
        self.agents_circuits = [
            ParametricQuantumCircuit(self.num_qubits_per_agent) for _ in range(self.num_agents)
        ]
        self.agents_qc_gates = [[] for _ in range(self.num_agents)]
        # Action space initialization
        for i, circuit in enumerate([self.total_qc, *self.agents_circuits]):
            oneq_gates = []
            rot_gates = [
                circuit.add_parametric_RX_gate,
                circuit.add_parametric_RY_gate
            ]
            twoq_gates = ['CNOT', 'flip_CNOT']  # do not change or the code will collapse
            skip_action = ['pass']
            all_gates = oneq_gates + rot_gates + twoq_gates + skip_action
            if i == 0:
                self.oneq_gate_range = len(oneq_gates)
                self.rot_gate_range = sum(len(gate_arr) for gate_arr in [oneq_gates, rot_gates])
                self.twoq_gate_range = sum(len(gate_arr) for gate_arr in [oneq_gates, rot_gates, twoq_gates])
                self.skip_gate_range = sum(
                    len(gate_arr) for gate_arr in [oneq_gates, rot_gates, twoq_gates, skip_action])
                self.total_qc_gates = all_gates
                self.max_action = len(all_gates[:-1]) * self.num_qubits_per_agent + 1
            else:
                self.agents_qc_gates[i - 1] = all_gates

        # Hadamard initialization
        for i in range(self.num_qubits):
            self.total_qc.add_H_gate(i)
        for agent_circ in self.agents_circuits:
            for i in range(self.num_qubits_per_agent):
                agent_circ.add_H_gate(i)
        # Reset states
        for agent_state in self.agent_states:
            agent_state.set_zero_state()
        self.state.set_zero_state()
        # Some miscellaneous parameters
        self.action_history = np.zeros(shape=(self.max_episode_len, self.num_agents))
        self.agents_action_history = np.zeros(
            shape=(self.num_agents, self.max_episode_len))  # has no use for now (debug)
        self.param_counter = 0  # to understand the size of ALL parameter vector
        self.optimized_params = None
        self.agents_optimized_params = None
        self.parameter_indices = []
        self.approx_ratio = 0
        self.test_ar = 0

        self.step_idx = 0
        initial_state = self.get_obs()
        info = {'circuit': 'none'}
        return initial_state, info

    def step(self, actions):
        """
        One environment step. Receives actions, produces the consequences.
        :param actions: [a0,...,aN]
        :return: obs vector, reward, terminated flag, done flag, info
        """
        if not self.independent:
            if self.step_idx % (self.num_qubits / self.num_agents) == 0:  # Layer-wise kinda optimization
                self.step_params = []  # at each %%self.num_qubits / self.num_agents%% step, create another VARIABLE parameter
        if self.debug:
            print(f'/' * 40)
            print(f'Received actions {actions}')
        step_has_rotations = False
        # Process actions, apply gates to circuit(s)
        for agent_idx, action in enumerate(actions):
            selected_gate = action // self.num_qubits_per_agent  # convert action token to actual gate
            agent_qubit_index = action % self.num_qubits_per_agent
            qubit_index = agent_qubit_index + agent_idx * self.num_qubits_per_agent
            add_total_qc_gate = self.total_qc_gates[selected_gate]  # select gate for total circuit
            add_agent_qc_gate = self.agents_qc_gates[agent_idx][selected_gate]  # and for the agents

            # boolean flags for gate selection
            selected_oneq = selected_gate < self.oneq_gate_range
            selected_rot = self.oneq_gate_range <= selected_gate < self.rot_gate_range
            selected_twoq = self.rot_gate_range <= selected_gate < self.twoq_gate_range
            selected_skip = self.twoq_gate_range <= selected_gate < self.skip_gate_range

            if selected_oneq:
                add_total_qc_gate(index=qubit_index)
                add_agent_qc_gate(index=agent_qubit_index)
                if self.debug:
                    print(f'1Q gate at qubit {qubit_index} ({agent_idx}/{agent_qubit_index})'
                          f', name:{add_total_qc_gate.__name__}')
            elif selected_rot:
                add_total_qc_gate(index=qubit_index, angle=7)
                add_agent_qc_gate(index=agent_qubit_index, angle=7)
                if not self.independent:
                    self.step_params.append(self.param_counter)
                self.param_counter += 1
                step_has_rotations = True
                if self.debug:
                    print(f'Rotation at qubit {qubit_index} ({agent_idx}/{agent_qubit_index}),'
                          f' name:{add_total_qc_gate.__name__}, '
                          f'{self.param_counter} total params')
            elif selected_twoq:
                # FIXME: HARDCODE!
                boundary = False
                if selected_gate == self.rot_gate_range:  # cnot regular
                    qubit_indices = (qubit_index, qubit_index + 1)
                    agent_qubit_indices = (agent_qubit_index, agent_qubit_index + 1)
                    if qubit_index == self.num_qubits - 1:
                        boundary = True
                        qubit_indices = (self.num_qubits - 1, 0)
                elif selected_gate == self.rot_gate_range + 1:  # cnot flip
                    qubit_indices = (qubit_index + 1, qubit_index)
                    agent_qubit_indices = (agent_qubit_index + 1, agent_qubit_index)
                    if qubit_index == self.num_qubits - 1:
                        boundary = True
                        qubit_indices = (0, self.num_qubits - 1)
                total_control, total_target = qubit_indices
                self.total_qc.add_CNOT_gate(control=total_control, target=total_target)
                # if self.debug:
                #     print(
                #         f'CNOT for total circuit, qubits: control {total_control}, target {total_target}, {"is" if boundary else 'not'} boundary')
                if agent_qubit_index != self.num_qubits_per_agent - 1:  # skip boundary case
                    control, target = agent_qubit_indices
                    if self.debug:
                        print(f'CNOT for agent {agent_idx}, qubits: control {control}, target {target}')
                    self.agents_circuits[agent_idx].add_CNOT_gate(control=control,
                                                                  target=target)
                else:
                    if self.debug:
                        print(f'CNOT boundary case skipped for agent {agent_idx}')

            elif selected_skip:  # pass
                if self.debug:
                    print(f'Gate skip, token {action}, skip idx {selected_gate}, agent {agent_idx}')
                continue
            self.action_history[self.step_idx][agent_idx] = action

        if not self.independent:
            # "layer"-wise structure
            if self.step_idx % (
                    self.num_qubits / self.num_agents) == 0 and step_has_rotations:  # append params by groups for combined optimization
                self.parameter_indices.append(self.step_params)
        param_count = self.total_qc.get_parameter_count()
        if param_count != 0:
            ratios = []
            for i in range(3):
                if not self.independent:
                    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=(3, len(self.parameter_indices)))
                else:
                    init_params = np.random.uniform(low=0.0, high=2 * np.pi,
                                                    size=(3, self.total_qc.get_parameter_count()))
                appr = self.optimize_angles(init_params=init_params[i])
                ratios.append(appr)
            self.approx_ratio = max(ratios)
        else:  # no rotational gates
            self.approx_ratio = self.optimize_angles(init_params=None, no_angles=True)

        for agent, agent_state in zip(self.agents_circuits, self.agent_states):
            agent.update_quantum_state(agent_state)

        # after optimization, get the final observations and reward
        observations = self.get_obs()
        reward = (2 * self.approx_ratio - 1) - 0.005 * self.step_idx  # simple reward for testing purposes
        # reward += np.log(1-self.approx_ratio*0.99)**2*np.sign(self.approx_ratio)
        # reward = 0
        if self.debug:
            print(f'\nAR: {self.approx_ratio:.2f}')
            print(f'REWARD: {reward:.2f}')
        if self.approx_ratio > self.ar_threshold:
            self.ar_found += 1
            # self.test_ar = self.optimize_angles(test=True)
            if self.debug:
                # print(f'THRESHOLD REACHED, TEST AR: {self.test_ar:.2f}')
                print(f'Threshold reached')

        done = False
        if self.step_idx == self.max_episode_len - 1 or self.approx_ratio > self.ar_threshold:
            done = True
            # reward = (2 * self.approx_ratio - 1) - 0.005 * self.step_idx  # simple reward for testing purposes
        info = {
            'circuit': self.total_qc,
            'action_history': self.action_history
        }
        self.step_idx += 1

        # JUST IN CASE, however, Qulacs should forget the previous state during invocation of update_quantum_state() method
        self.state.set_zero_state()
        for agent_state in self.agent_states:
            agent_state.set_zero_state()
        return observations, reward, False, done, info

    def get_metric(self):
        statevector = self.state.get_vector()
        self.expval = np.real(np.vdot(statevector, np.dot(self.hamiltonian, statevector)))
        self.ar = self.expval / self.min_energy
        return self.ar

    def optimize_angles(self,
                        init_params,
                        no_angles=False
                        ):
        """
        Speaks for itself, angle optimization
        :param test: use test set instead of train
        :param no_angles: if circuit has no parameters, get A.R. anyway
        :return:
        """
        if self.debug:
            print(f'Optimization started...')
            print(f'Total circuit params {self.param_counter}')
        if no_angles:
            if self.debug:
                print(f'No parameters encountered')
            self.total_qc.update_quantum_state(self.state)
            optimized_energy = self.get_metric()
        else:
            if self.debug:
                print(f'Init variable params: {init_params}')
                print(f'Init energy: {self.get_metric():.2f}')
            optimization_result = minimize(fun=self.update_angles,
                                           x0=init_params,
                                           # bounds= [(0,np.pi/4) for _ in range(3)],
                                           method='COBYLA',
                                           )
            optimized_params = optimization_result.x
            optimized_energy = -optimization_result.fun
            self.found_energy = self.expval
            if self.debug:
                print(f'Optimized variable params: {optimized_params}')
                print(
                    f'Optimized circuit params: '
                    f'{[self.total_qc.get_parameter(ind) for ind in range(self.total_qc.get_parameter_count())]}')
                print(f'Optimized metric: {self.found_energy:.2f}')
                print(f'Found energy: {self.expval:.2f}')
                print(f'Target energy: {self.min_energy:.2f}')

        # print(f'AR: {self.ar:.2f}, FID: {self.overlap:.2f}')
        return optimized_energy

    def test_circuit(self, path):
        """
        Load prepared circuit in .pickle format, optimize it and check performance
        :param path: path to .pickle file
        :return:
        """
        self.reset()
        with open(path, 'rb') as file:
            circuit = pickle.load(file)
        self.total_qc = circuit
        print(f'Parameters: {[self.total_qc.get_parameter(i) for i in range(self.total_qc.get_parameter_count())]}')
        # qc_params = np.array([self.total_qc.get_parameter(i) for i in range(self.total_qc.get_parameter_count())])
        # qc_params = np.round(qc_params, 2)
        # uniq_values, counts = np.unique(qc_params, return_counts=True)
        # counts = list(accumulate(counts, lambda a, b: a + b))
        # self.parameter_indices = np.split(np.array([i for i in range(self.total_qc.get_parameter_count())]),
        #                                   counts[:-1])
        self.total_qc.update_quantum_state(self.state)
        statevector = self.state.get_vector()
        # self.test_ar = self.get_metric()
        self.test_ar = np.real(np.conj(statevector).T @ self.hamiltonian @ statevector)
        self.test_ar = self.test_ar / self.min_energy

        # print(f'TEST AR: {self.test_ar:.2f}')
        _, _, ground_state, _ = self.get_min_max_energy(get_state=True)
        print(f'TEST GROUND STATE FIDELITY: {np.abs(np.vdot(statevector, ground_state)) ** 2}')
        # print(f'AR VALUES FOR EACH TEST INSTANCE: {self.test_ar_array}')

    def update_angles(self, x):
        """
        Parameter update function used in SciPy minimize loop.
        :param x: solution vector
        :param maxcut_obs:
        :param max_e:
        :param min_e:
        :return: negative AR
        """
        if not self.independent:
            # LAYER-WISE
            for i, param_set in enumerate(self.parameter_indices):
                for param_idx in param_set:
                    self.total_qc.set_parameter(param_idx, parameter=x[i])
        else:
            # INDEPENDENT
            for i in range(self.total_qc.get_parameter_count()):
                self.total_qc.set_parameter(i, parameter=x[i])
        self.total_qc.update_quantum_state(self.state)
        # ar = self.get_ar(maxcut_obs=maxcut_obs, max_e=max_e, min_e=min_e)
        expval = self.get_metric()
        self.state.set_zero_state()
        return -expval

    @staticmethod
    def _float_flatten(vector):
        vec = np.array([vector.real, vector.imag])
        vec = vec.flatten()
        vec = np.array(vec, dtype=float)
        return vec

    def get_obs(self):
        """
        Produce an array of [obs1,obs2,...,obsN]
        :return:
        """
        obs = [self._float_flatten(agent_state.get_vector()) for agent_state in self.agent_states]
        if self.debug:
            print(f'get_obs(): obs vector {obs}')
        return obs

    def get_state(self):
        """
        Get full quantum state
        :return:
        """
        stv = self.state.get_vector()
        stv = self._float_flatten(stv)
        return stv

    def get_avail_actions(self):
        """
        External method for QMIX
        :return:
        """
        all_agents_possible_actions = np.zeros(shape=(self.num_agents, self.max_action), dtype=int)
        action_mask = np.array([1 for _ in range(self.max_action)])
        all_agents_possible_actions[:] = action_mask
        return all_agents_possible_actions

    def get_maxcut_observable(self, graph):
        """
        Converts graph to Qulacs observable
        :param graph: nx graph
        :return: Observable object
        """
        cost_obs = Observable(self.num_qubits)
        for edge in graph.edges():
            i, j = edge
            cost_obs.add_operator(PauliOperator(f'Z {i} Z {j}', -0.5))  # QAOA convention
            cost_obs.add_operator(PauliOperator(f'', 0.5))
        return cost_obs

    def get_env_info(self):
        """
        External method for QMIX
        :return:
        """
        return {
            'n_agents': self.num_agents,
            'obs_shape': len(self.get_obs()[0]),
            'state_shape': 2 ** (self.num_qubits + 1),  # +1 stands for complex number representation
            'n_actions': self.max_action,
            'episode_limit': self.max_episode_len
        }

    # def get_graph_params(self, maxcut_obs_array):
    #     """
    #     Extracts max/min energy from each of MaxCut observables in the given array
    #     :param maxcut_obs_array: [obs1,obs2,...,obsN]
    #     :return: [min E1, minE2,...,minEN], [max E1,max E2,...,maxEN]
    #     """
    #     energies = np.zeros((2, len(maxcut_obs_array)))
    #     for i, maxcut_obs in enumerate(maxcut_obs_array):
    #         hamiltonian_maxcut = maxcut_obs.get_matrix().toarray()
    #         eigvals, eigvecs = eigh(hamiltonian_maxcut)
    #         min_energy, max_energy = eigvals[0], eigvals[-1]
    #         energies[0][i] = min_energy
    #         energies[1][i] = max_energy
    #         # max_energy_vector = eigvecs[:, -1]
    #     return energies[0], energies[1]

    def get_min_max_energy(self, get_state=False):
        """"""
        eigvals, eigvecs = eigh(self.hamiltonian)
        min_energy, max_energy = np.min(eigvals), np.max(eigvals)
        self.ground_state = eigvecs[:, np.argmin(eigvals)]
        self.gap_value = eigvals[1] - eigvals[0]
        if get_state:
            return min_energy, max_energy, self.ground_state, 0
        else:
            return min_energy, max_energy


if __name__ == '__main__':
    num_qubits = 8
    num_agents = 2
    max_ep_len = 20
    env = VQE_QAS(num_qubits,
                  num_agents=num_agents,
                  max_episode_len=max_ep_len,
                  debug=True
                  )
    env.reset(seed=42)
    # SIMPLE TEST
    template_actions = [[i for _ in range(num_agents)] for i in range(0, env.max_action, 1)]
    for action in template_actions:
        allowed_actions = env.get_avail_actions()
        obs, reward, terminated, done, info = env.step(action)

    # load circuit, inspect it

    # env.test_circuit('name.pickle')
