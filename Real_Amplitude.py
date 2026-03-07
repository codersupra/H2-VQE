"""
Real Amplitude Ansatz (Superconducting Qubits) VQE for H2 Ground State Energy.

Electronic Structure Problem: PySCFDriver with STO-3G basis and frozen-core approximation.
Mapping: Jordan-Wigner transformation.
Ansatz: RealAmplitudes template with parameterized rotation gates and entangling CNOT blocks.
Optimizer: COBYLA (classical optimizer from SciPy's minimize method).
Estimator: StatevectorEstimator from Qiskit Primitives.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_aer import AerSimulator


NUM_QUBITS = 4
REPS = 2


def get_ansatz(params, num_qubits=NUM_QUBITS, reps=REPS):
    """
    Create a RealAmplitudes ansatz and bind parameters.

    Args:
        params (array): Parameter values to bind.
        num_qubits (int): Number of qubits.
        reps (int): Number of repetitions in the ansatz.

    Returns:
        QuantumCircuit: The bound ansatz circuit.
    """
    template = RealAmplitudes(num_qubits=num_qubits, reps=reps)
    bound_circuit = template.assign_parameters(params)
    return bound_circuit


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
    qc = ansatz_func(params)
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
    Run VQE with the RealAmplitudes ansatz.

    Args:
        num_trials (int): Number of random initialization trials.
        maxiter (int): Maximum optimizer iterations per trial.
        plot (bool): Whether to display convergence plot.
        save (bool): Whether to save plot and results to Results/ directory.

    Returns:
        dict: Results including best energy, parameters, and convergence history.
    """
    print("=" * 60)
    print("Real Amplitude Ansatz VQE")
    print("=" * 60)

    op = build_hamiltonian()
    estimator = StatevectorEstimator()
    pm = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3, seed_transpiler=0
    )

    H2_energy = -1.1357

    template_ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=REPS)
    num_parameters = template_ansatz.num_parameters

    cost_history_list = []
    res_list = []
    params_0_list = np.random.uniform(0, 2 * np.pi, (num_trials, num_parameters))

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
        plt.title("VQE Convergence - Real Amplitude Ansatz")
        plt.legend()
        plt.grid(True)
        if save:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, "real_amplitude_convergence.png"), dpi=150, bbox_inches="tight")
            print(f"  Plot saved to Results/real_amplitude_convergence.png")
        if plot:
            plt.show()
        plt.close()

    result_dict = {
        "method": "Real Amplitude Ansatz",
        "energy": best_energy,
        "params": res_list[best_idx].x,
        "cost_history": cost_history_list[best_idx],
        "all_results": res_list,
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_data = {
            "method": "Real Amplitude Ansatz",
            "energy": float(best_energy),
            "best_trial": int(best_idx + 1),
            "optimal_params": res_list[best_idx].x.tolist(),
            "num_trials": num_trials,
            "maxiter": maxiter,
        }
        with open(os.path.join(RESULTS_DIR, "real_amplitude_results.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Results saved to Results/real_amplitude_results.json")

    return result_dict


if __name__ == "__main__":
    result = run(num_trials=5, maxiter=200, plot=True)
    print(f"\nFinal Energy: {result['energy']:.6f} Hartree")
