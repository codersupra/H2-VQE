"""
Classical Variational Principle for H2 Ground State Energy.

Uses a trial wavefunction (product of 1s hydrogenic orbitals) and
minimizes the energy with respect to a variational parameter alpha.
All integrals are computed numerically using scipy.integrate.quad.
"""

import os
import json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def psi(r, alpha, R):
    """Trial wavefunction: sum of 1s hydrogenic orbitals centered at +/- R/2."""
    return np.exp(-alpha * np.abs(r + R / 2)) + np.exp(-alpha * np.abs(r - R / 2))


def normalization(alpha, R):
    """Normalization integral: integral of |psi(r)|^2 dr."""
    integrand = lambda r: np.abs(psi(r, alpha, R)) ** 2
    result, _ = quad(integrand, -10, 10)
    return result


def kinetic_energy(alpha, R):
    """Kinetic energy: -1/2 integral of psi * d^2(psi)/dr^2 dr."""

    def d2psi(r):
        return (
            -alpha ** 2 * np.exp(-alpha * np.abs(r + R / 2))
            - alpha ** 2 * np.exp(-alpha * np.abs(r - R / 2))
        )

    integrand = lambda r: psi(r, alpha, R) * d2psi(r)
    result, _ = quad(integrand, -10, 10)
    return -0.5 * result / normalization(alpha, R)


def potential_energy(alpha, R):
    """Electron-nucleus potential energy."""
    integrand = lambda r: (
        np.abs(psi(r, alpha, R)) ** 2
        * (-1 / np.abs(r - R / 2) - 1 / np.abs(r + R / 2))
    )
    result, _ = quad(integrand, -10, 10)
    return result / normalization(alpha, R)


def repulsion(R):
    """Internuclear repulsion energy: 1/R (in atomic units)."""
    return 1 / R


def total_energy(alpha, R):
    """Total energy E(alpha) = T + V + V_nn."""
    T = kinetic_energy(alpha, R)
    V = potential_energy(alpha, R)
    Vnn = repulsion(R)
    return T + V + Vnn


def run(plot=False, save=True):
    """
    Run classical variational principle calculation for H2.

    Args:
        plot (bool): Not used (single parameter optimization).
        save (bool): Whether to save results to Results/ directory.

    Returns:
        dict: Results including minimum energy and optimal alpha.
    """
    print("=" * 60)
    print("Classical Variational Principle")
    print("=" * 60)

    # Internuclear distance in Bohr (0.735 Angstrom ~ 1.388 Bohr)
    R = 1.388

    result = minimize_scalar(
        lambda alpha: total_energy(alpha, R), bounds=(0.1, 3.0), method="bounded"
    )

    energy = result.fun
    alpha = result.x
    print(f"\n  Ground State Energy: {energy:.6f} Hartree")
    print(f"  Optimal alpha: {alpha:.6f}")

    result_dict = {
        "method": "Classical Variational Principle",
        "energy": energy,
        "alpha": alpha,
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_data = {
            "method": "Classical Variational Principle",
            "energy": float(energy),
            "optimal_alpha": float(alpha),
            "internuclear_distance_bohr": R,
        }
        with open(os.path.join(RESULTS_DIR, "classical_variational_results.json"), "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Results saved to Results/classical_variational_results.json")

    return result_dict


if __name__ == "__main__":
    result = run()
    print(f"\nFinal Energy: {result['energy']:.6f} Hartree")
