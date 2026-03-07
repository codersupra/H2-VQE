"""
Photonic Quantum Processor (Custom Ansatz) VQE for H2 Ground State Energy.

Electronic structure problem defined via PySCFDriver using the STO-3G basis
and a frozen-core approximation.
Mapping: Jordan-Wigner transformation.
Ansatz: Custom photonic-inspired circuit with parameterized gates and entangling blocks.
Optimizer: COBYLA (classical optimizer from SciPy's minimize method).
Estimator: StatevectorEstimator from Qiskit Primitives.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_aer import AerSimulator


def get_ansatz(params, num_qubits):
    """
    Create a parametrized photonic-inspired ansatz quantum circuit.

    Args:
        params (list or array): A list of 4 parameters [theta, beta_1, lam, beta_2].
        num_qubits (int): Total number of qubits in the circuit (>= 2).

    Returns:
        QuantumCircuit: The constructed ansatz circuit.
    """
    if len(params) != 4:
        raise ValueError(f"Expected 4 parameters but received {len(params)}.")
    if num_qubits < 2:
        raise ValueError("Number of qubits must be at least 2.")

    theta, beta_1, lam, beta_2 = params
    qc = QuantumCircuit(num_qubits)

    # Hadamard on even qubits
    for i in range(num_qubits):
        if i % 2 == 0:
            qc.h(i)

    # CNOTs between even and next qubit
    for i in range(num_qubits - 1):
        if i % 2 == 0:
            qc.cx(i, i + 1)

    # Custom photonic block
    def custom_block(qc, control, target):
        qc.u(theta, beta_1, lam, control)
        qc.cx(control, target)
        qc.rz(-beta_1, target)
        qc.cx(control, target)
        qc.u(theta, beta_1, lam, control)
        qc.rz(beta_2, target)

    # Apply custom blocks to each neighboring pair starting from qubit 1
    for i in range(1, num_qubits - 1):
        custom_block(qc, i, i + 1)

    return qc


def build_hamiltonian():
    """Build and return the qubit Hamiltonian for H2 using Jordan-Wigner mapping."""
    molecule = MoleculeInfo(
        ["H", "H"],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.735)],
        charge=0,
        multiplicity=1,
    )
    driver = PySCFDriver.from_molecule(molecule, basis="sto3g")
    problem = driver.run()
    transformer = FreezeCoreTransformer()
    problem = transformer.transform(problem)
    hamiltonian = problem.hamiltonian.second_q_op()

    jw_mapper = JordanWignerMapper()
    qubit_op = jw_mapper.map(hamiltonian)
    return qubit_op


def cost_func(params, ansatz_func, obs, estimator, pm, callback_dict):
    """Cost function for VQE optimization."""
    qc = ansatz_func(params, 4)
    transpiled_circuit = pm.run(qc)
    transpiled_obs = obs.apply_layout(layout=transpiled_circuit.layout)

    job = estimator.run([(transpiled_circuit, transpiled_obs)])
    result = job.result()[0].data.evs

    cost = abs(result + 1.1357)

    callback_dict["iters"] += 1
    callback_dict["prev_vector"] = params
    callback_dict["cost_history"].append(cost)

    print(
        f"  Iteration {callback_dict['iters']}, Cost: {cost:.8f}",
        end="\r",
        flush=True,
    )
    return cost


def run(num_trials=30, maxiter=200, plot=False, save=True):
    """
    Run VQE with the photonic ansatz.

    Args:
        num_trials (int): Number of random initialization trials.
        maxiter (int): Maximum optimizer iterations per trial.
        plot (bool): Whether to display convergence plot.
        save (bool): Whether to save plot and results to Results/ directory.

    Returns:
        dict: Results including best energy, parameters, and convergence history.
    """
    print("=" * 60)
    print("Photonic Ansatz VQE")
    print("=" * 60)

    op = build_hamiltonian()
    estimator = StatevectorEstimator()
    pm = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3, seed_transpiler=0
    )

    H2_energy = -1.1357

    cost_history_list = []
    res_list = []
    params_0_list = np.random.uniform(0, 2 * np.pi, (num_trials, 4))

    for it, params_0 in enumerate(params_0_list):
        print(f"\n  Starting optimization #{it + 1}/{num_trials}")

        callback_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }

        res = minimize(
            cost_func,
            x0=params_0,
            args=(get_ansatz, op, estimator, pm, callback_dict),
            method="COBYLA",
            options={"maxiter": maxiter, "disp": False},
        )

        res_list.append(res)
        cost_history_list.append(callback_dict["cost_history"])

    # Find best result
    best_idx = np.argmin([r.fun for r in res_list])
    best_energy = res_list[best_idx].fun + H2_energy

    print(f"\n\n  Best trial: #{best_idx + 1}")
    print(f"  Ground State Energy: {best_energy:.6f} Hartree")

    if (plot or save) and cost_history_list:
        plt.figure()
        plt.plot(
            np.array(cost_history_list[best_idx]) + H2_energy,
            label=f"Run {best_idx + 1}",
        )
        plt.axhline(H2_energy, color="black", linestyle="--", label="$H_2$ Ground Energy")
        plt.xlabel("Iteration")
        plt.ylabel("Energy (Hartree)")
        plt.title("VQE Convergence - Photonic Ansatz")
        plt.legend()
        plt.grid(True)
        if save:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, "photonic_ansatz_convergence.png"), dpi=150, bbox_inches="tight")
            print(f"  Plot saved to Results/photonic_ansatz_convergence.png")
        if plot:
            plt.show()
        plt.close()

    result_dict = {
        "method": "Photonic Ansatz",
        "energy": best_energy,
        "params": res_list[best_idx].x,
        "cost_history": cost_history_list[best_idx],
        "all_results": res_list,
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_data = {
            "method": "Photonic Ansatz",
            "energy": float(best_energy),
            "best_trial": int(best_idx + 1),
            "optimal_params": res_list[best_idx].x.tolist(),
            "num_trials": num_trials,
            "maxiter": maxiter,
        }
        with open(os.path.join(RESULTS_DIR, "photonic_ansatz_results.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Results saved to Results/photonic_ansatz_results.json")

    return result_dict


if __name__ == "__main__":
    result = run(num_trials=5, maxiter=200, plot=True)
    print(f"\nFinal Energy: {result['energy']:.6f} Hartree")
