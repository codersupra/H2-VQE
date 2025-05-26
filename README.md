# H2-VQE
This is a project for comparing the ground state energies of H2 molecule obtained using different quantum computing optimization techniques. I have also added the comparison with the results using classical scipy minimizer. Currently only VQE (Variational Quantum Eigensolver) is used, other optimization techniques will be added in future.

## Methods:
  ### 1. Photonic Quantum Processor (Custom Ansatz)

      Electronic structure problem defined via PySCFDriver using the STO-3G basis and a frozen-core approximation.

      Mapping: Jordan-Wigner transformation to convert fermionic operators to qubit operators.
      
      Ansatz: Custom photonic-inspired circuit with parameterized gates and entangling blocks.
      
      Optimizer: COBYLA (classical optimizer from SciPy's minimize method).
      
      Estimator: StatevectorEstimator from Qiskit Primitives for expectation value calculation.
  ### 2. Real Amplitude Ansatz (Superconducting Qubits)

      Electronic Structure Problem: Defined using the PySCFDriver with the STO-3G basis set and a frozen-core approximation to reduce the active space and improve computational efficiency.

      Mapping Scheme: Applied the Jordan-Wigner transformation to convert the second-quantized fermionic Hamiltonian into a qubit-based representation compatible with quantum circuits.
      
      Ansatz Design: Employed a custom photonic-inspired ansatz built using Qiskit’s RealAmplitudes template, incorporating parameterized single-qubit rotation gates and entangling CNOT blocks across four qubits to capture electronic correlations.
      
      Optimization: Utilized the COBYLA algorithm from SciPy's minimize method to iteratively adjust the ansatz parameters and minimize the expectation value of the Hamiltonian.
      
      Estimator: Used Qiskit's StatevectorEstimator to compute expectation values of the Hamiltonian with respect to the variational quantum state, simulating ideal (noiseless) quantum measurement outcomes.
  ### 3. UCCSD Ansatz (Superconducting Qubits)

      Electronic structure problem defined via PySCFDriver using the STO-3G basis.

      Mapping: Jordan-Wigner transformation to convert fermionic operators to qubit operators.
      
      Ansatz: UCCSD with Hartree-Fock initial state.
      
      Optimizer: SLSQP (classical optimizer).
      
      Estimator: Qiskit's built-in Estimator for energy evaluation.
  

## Results:
Circuit.png shows the photonic ansatz
Fig1.png in the results folder shows the convergence curve for the photonic processor ansatz.

1. Photonic Ansatz: -1.1356 Hartrees
2. Real Amplitude Ansatz: -1.13568 Hartrees
3. UCCSD Ansatz: -1.1373 Hartrees

## Reference:
  Ghavami, Badie & Mirmasuodi, Forouzan. (2025). Ground State Energy of Helium Using a Four-Qubit Photonic Processor with the Variational Quantum Eigensolver (VQE). 10.48550/arXiv.2504.07568. 
