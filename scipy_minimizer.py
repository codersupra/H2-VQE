"""
Classical Scipy Minimizer VQE for H2 Ground State Energy.

Uses a 2-qubit matrix representation with Pauli operators and a simple
Ry rotation ansatz. Optimization done purely classically via SciPy's minimize.
No quantum circuit simulator is used - all operations are matrix multiplications.
"""

import os
import json
import numpy as np
from scipy.optimize import minimize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def kron2(a, b):
    """Kronecker product of two matrices."""
    return np.kron(a, b)


# H2 molecular Hamiltonian in STO-3G basis (Jordan-Wigner, 2-qubit form)
H2_HAMILTONIAN = (
    -1.0523732 * kron2(I, I)
    + 0.39793742 * kron2(Z, I)
    - 0.39793742 * kron2(I, Z)
    - 0.0112801 * kron2(Z, Z)
    + 0.18093119 * kron2(X, X)
)


def ansatz(theta):
    """
    Single-parameter Ry ansatz on both qubits.

    Args:
        theta (array): Two rotation angles [theta_0, theta_1].

    Returns:
        ndarray: The 4x1 statevector.
    """
    ry0 = np.array([
        [np.cos(theta[0] / 2), -np.sin(theta[0] / 2)],
        [np.sin(theta[0] / 2), np.cos(theta[0] / 2)],
    ])
    ry1 = np.array([
        [np.cos(theta[1] / 2), -np.sin(theta[1] / 2)],
        [np.sin(theta[1] / 2), np.cos(theta[1] / 2)],
    ])
    state = np.kron(ry0 @ np.array([[1], [0]]), ry1 @ np.array([[1], [0]]))
    return state


def cost(theta):
    """Cost function: <psi(theta)| H |psi(theta)>."""
    psi = ansatz(theta)
    energy = np.vdot(psi, H2_HAMILTONIAN @ psi).real
    return energy


def run(plot=False, save=True):
    """
    Run classical scipy-based VQE for H2.

    Args:
        plot (bool): Not used (single-shot optimization).
        save (bool): Whether to save results to Results/ directory.

    Returns:
        dict: Results including minimum energy and optimal parameters.
    """
    print("=" * 60)
    print("Scipy Minimizer (Classical Matrix VQE)")
    print("=" * 60)

    theta0 = [0.0, 0.0]
    res = minimize(cost, theta0, method="COBYLA")

    energy = res.fun
    print(f"\n  Ground State Energy: {energy:.6f} Hartree")
    print(f"  Optimal parameters: {res.x}")

    result_dict = {
        "method": "Scipy Minimizer",
        "energy": energy,
        "params": res.x,
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_data = {
            "method": "Scipy Minimizer",
            "energy": float(energy),
            "optimal_params": res.x.tolist(),
        }
        with open(os.path.join(RESULTS_DIR, "scipy_minimizer_results.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Results saved to Results/scipy_minimizer_results.json")

    return result_dict


if __name__ == "__main__":
    result = run()
    print(f"\nFinal Energy: {result['energy']:.6f} Hartree")
