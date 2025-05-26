# H2-VQE
This is a project for comparing the ground state energies of H2 molecule obtained using different quantum computing optimization techniques. I have also added the comparison with the results using classical scipy minimizer. Currently only VQE (Variational Quantum Eigensolver) is used, other optimization techniques will be added in future.

## Methods:
  ### 1. Photonic Quantum Processor (Custom Ansatz)

      Mapping: Jordan-Wigner transformation to convert fermionic operators to qubit operators.
      
      Ansatz: UCCSD with Hartree-Fock initial state.
      
      Optimizer: SLSQP (classical optimizer).
      
      Estimator: Qiskit's built-in Estimator for energy evaluation.
  

## Results:
Fig1.jpg shows the convergence curve for the photonic processor ansatz.

1. Photonic Ansatz: -1.1356 Hartrees
2. Real Amplitude Ansatz: -1.13568 Hartrees
3. UCCSD Ansatz: -1.1373 Hartrees
