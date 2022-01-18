# spquantum
Simple Schrödinger-Poisson solver for the simulation of quantum devices. The package also provides a framework to simulate time-independent and time-dependent (given time-varying potentials) Schrödinger equations in 1D, 2D, and 3D. The main framework uses finite difference schemes to solve for the energies (eigenstates) and their corresponding wavefunctions (eigenfunctions). Approaches to solve multilevel systems using density matrix methods and the Wigner function are also available. For time-dependent problems there two propagation schemes are provided: 
- Crank-Nicholson
- Leapfrog
