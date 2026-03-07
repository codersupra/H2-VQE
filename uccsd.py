"""
UCCSD Ansatz (Superconducting Qubits) VQE for H2 Ground State Energy.

Electronic structure problem defined via PySCFDriver using the STO-3G basis.
Mapping: Jordan-Wigner transformation.
Ansatz: UCCSD with Hartree-Fock initial state.
Optimizer: SLSQP (classical optimizer via SciPy).
Estimator: StatevectorEstimator from Qiskit Primitives.
"""

import os
import json
import numpy as np
from scipy.optimize import minimize

from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from qiskit_aer import AerSimulator

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def run(plot=False, save=True):
    """
    Run VQE with UCCSD ansatz and Hartree-Fock initial state.

    Args:
        plot (bool): Not used for this method (no iterative convergence to plot).
        save (bool): Whether to save results to Results/ directory.

    Returns:
        dict: Results including ground state energy.
    """
    print("=" * 60)
    print("UCCSD Ansatz VQE")
    print("=" * 60)

    # Define molecule and run driver
    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
    problem = driver.run()

    # Jordan-Wigner mapping
    mapper = JordanWignerMapper()
    hamiltonian = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(hamiltonian)

    # UCCSD ansatz with Hartree-Fock initial state
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )

    num_params = ansatz.num_parameters
    estimator = StatevectorEstimator()
    pm = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3, seed_transpiler=0
    )

    # Nuclear repulsion energy (constant offset)
    nuclear_repulsion = problem.nuclear_repulsion_energy

    def cost_func(params):
        bound_circuit = ansatz.assign_parameters(params)
        transpiled = pm.run(bound_circuit)
        transpiled_op = qubit_op.apply_layout(layout=transpiled.layout)
        job = estimator.run([(transpiled, transpiled_op)])
        electronic_energy = job.result()[0].data.evs
        return float(electronic_energy)

    # Optimize
    x0 = np.zeros(num_params)
    res = minimize(cost_func, x0, method="SLSQP", options={"maxiter": 200, "disp": False})

    electronic_energy = res.fun
    total_energy = electronic_energy + nuclear_repulsion
    print(f"\n  Electronic Energy:   {electronic_energy:.6f} Hartree")
    print(f"  Nuclear Repulsion:   {nuclear_repulsion:.6f} Hartree")
    print(f"  Ground State Energy: {total_energy:.6f} Hartree")

    result_dict = {
        "method": "UCCSD Ansatz",
        "energy": total_energy,
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_data = {
            "method": "UCCSD Ansatz",
            "energy": float(total_energy),
            "electronic_energy": float(electronic_energy),
            "nuclear_repulsion": float(nuclear_repulsion),
        }
        with open(os.path.join(RESULTS_DIR, "uccsd_results.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Results saved to Results/uccsd_results.json")

    return result_dict


if __name__ == "__main__":
    result = run()
    print(f"\nFinal Energy: {result['energy']:.6f} Hartree")
