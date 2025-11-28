import numpy as np
from functools import reduce
from numpy.linalg import eig


class SchwingerHamiltonian(object):
    def __init__(self, n_qubits, w=1, g=1, e0=0):
        self.n_qubits = n_qubits
        self.PauliX = np.array([[0., 1.], [1., 0.]])
        self.PauliY = np.array([[0., -1.j], [1.j, 0.]])
        self.PauliZ = np.array([[1., 0.], [0., -1.]])
        self.I = np.array([[1., 0.], [0., 1.]])

        self.matrix_dim = 2 ** self.n_qubits
        self.sp = (self.PauliX + 1j * self.PauliY) / 2
        self.sm = (self.PauliX - 1j * self.PauliY) / 2
        self.w = w
        self.g = g
        self.e0 = e0

    def make_hamiltonian(self,m, w=1, g=1, e0=0):

        N = self.n_qubits
        I = self.I
        Z = self.PauliZ
        d = 2 ** self.n_qubits

        sp = (self.PauliX + 1j * self.PauliY) / 2
        sm = (self.PauliX - 1j * self.PauliY) / 2
        term_1 = np.zeros([d, d], dtype=complex)
        for j in range(N - 1):
            op = reduce(np.kron, [self.I] * j + [sp, sm] + [I] * (N - j - 2))
            term_1 += op + op.conj().T

        term_2 = np.zeros([d, d], dtype=complex)
        for j in range(N):
            op = reduce(np.kron, [I] * j + [Z] + [I] * (N - j - 1))
            term_2 += (-1) ** (j + 1) * op

        term_3 = np.zeros([d, d], dtype=complex)
        for j in range(N):
            L_j = np.zeros([d, d], dtype=complex)
            for l in range(j + 1):
                op = self.PauliZ + (-1) ** (l + 1) * self.I
                op = reduce(np.kron, [self.I] * l + [op] + [self.I] * (N - l - 1))
                L_j += op
            L_j = e0 - L_j / 2
            term_3 += L_j @ L_j

        return w * term_1 + m / 2 * term_2 + g * term_3

    def prepare_ground_statevector_data(self, m_array):
        hamiltonians = [self.make_hamiltonian(m_) for m_ in m_array]
        base_statevector = [states[np.argmin(energies)] for energies, states in map(eig, hamiltonians)]
        return base_statevector

    def prepare_hamiltonians(self, m_array):
        return [self.make_hamiltonian(m_) for m_ in m_array]

if __name__=='__main__':
    # Step 1: Define the Hamiltonian
    n_qubits = 4  # Number of qubits
    m = 0.5  # Fermion mass
    w = 1.0  # Hopping strength
    g = 1.0  # Coupling constant

    shiwn = SchwingerHamiltonian(n_qubits=n_qubits)
    shiwn.prepare_ground_statevector_data(m_array=[-1, 0, 1])
