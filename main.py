"""
Main execution script for H2 Ground State Energy comparison.

Runs all VQE methods and classical approaches, then prints a comparison table.
Saves a summary JSON and comparison bar chart to the Results/ directory.
Methods:
  1. Photonic Ansatz (Custom Ansatz) VQE
  2. Real Amplitude Ansatz VQE
  3. UCCSD Ansatz VQE
  4. Scipy Minimizer (Classical Matrix VQE)
  5. Classical Variational Principle
"""

import os
import json
import matplotlib.pyplot as plt

import photonic_ansatz
import Real_Amplitude
import uccsd
import scipy_minimizer
import classical_variational

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def main():
    print("\n" + "#" * 60)
    print("  H2 Ground State Energy - Method Comparison")
    print("#" * 60 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    # 1. Photonic Ansatz VQE
    try:
        r = photonic_ansatz.run(num_trials=5, maxiter=200, plot=False, save=True)
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Photonic Ansatz failed: {e}")
        results.append({"method": "Photonic Ansatz", "energy": None})

    print()

    # 2. Real Amplitude Ansatz VQE
    try:
        r = Real_Amplitude.run(num_trials=5, maxiter=200, plot=False, save=True)
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Real Amplitude Ansatz failed: {e}")
        results.append({"method": "Real Amplitude Ansatz", "energy": None})

    print()

    # 3. UCCSD Ansatz VQE
    try:
        r = uccsd.run(save=True)
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] UCCSD Ansatz failed: {e}")
        results.append({"method": "UCCSD Ansatz", "energy": None})

    print()

    # 4. Scipy Minimizer (Classical Matrix VQE)
    try:
        r = scipy_minimizer.run(save=True)
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Scipy Minimizer failed: {e}")
        results.append({"method": "Scipy Minimizer", "energy": None})

    print()

    # 5. Classical Variational Principle
    try:
        r = classical_variational.run(save=True)
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Classical Variational failed: {e}")
        results.append({"method": "Classical Variational Principle", "energy": None})

    # Print comparison table
    ref_energy = -1.1373
    print("\n\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Method':<35} {'Energy (Hartree)':>20}")
    print("-" * 60)
    for r in results:
        energy = r["energy"]
        if energy is not None:
            print(f"  {r['method']:<35} {energy:>20.6f}")
        else:
            print(f"  {r['method']:<35} {'FAILED':>20}")
    print("-" * 60)
    print(f"  {'Reference (exact)':<35} {ref_energy:>20.4f}")
    print("=" * 60)

    # Save summary JSON
    summary = []
    for r in results:
        summary.append({
            "method": r["method"],
            "energy": float(r["energy"]) if r["energy"] is not None else None,
        })
    summary.append({"method": "Reference (exact)", "energy": ref_energy})

    with open(os.path.join(RESULTS_DIR, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to Results/comparison_summary.json")

    # Save comparison bar chart
    methods = [r["method"] for r in results if r["energy"] is not None]
    energies = [float(r["energy"]) for r in results if r["energy"] is not None]

    if methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, energies, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
        ax.axhline(ref_energy, color="black", linestyle="--", linewidth=1.5, label=f"Reference ({ref_energy:.4f} Ha)")
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title("H2 Ground State Energy - Method Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, energies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.002,
                f"{val:.4f}",
                ha="center", va="top", fontsize=9, fontweight="bold", color="white",
            )

        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "comparison_chart.png"), dpi=150, bbox_inches="tight")
        print(f"  Comparison chart saved to Results/comparison_chart.png")
        plt.close()


if __name__ == "__main__":
    main()
