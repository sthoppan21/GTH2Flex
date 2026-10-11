# GTH2Flex
Quantum Implementation of Hydrogen Producers Scheduling

# Theme 1: Quantum Impact
# Industry: Energy

# Problem:
Hydrogen producers face a scheduling challenge: electricity prices change throughout the day, but customers still need hydrogen delivered on time 

# Solution:
H2Flex demonstrates cost-aware hydrogen production scheduling and validates a Qiskit QAOA implementation against exact classical solutions. The small instance does not establish quantum advantage; it provides a reproducible foundation for testing richer production and storage constraints. 

# Outcome:

Conversion Logic Involved using QiSkit:
xi​=21−Zi​​
For a QUBO,
F(x)=c+∑iaixi+∑i<jbijxixj,F(x)=c+\sum_i a_ix_i+\sum_{i<j}b_{ij}x_ix_j,F(x)=c+i∑​ai​xi​+i<j∑​bij​xi​xj​,
the Ising coefficients are:
hi=−ai2−∑j≠ibij4,Jij=bij4.h_i=-\frac{a_i}{2}-\sum_{j\ne i}\frac{b_{ij}}{4}, \qquad J_{ij}=\frac{b_{ij}}{4}.hi​=−2ai​​−j=i∑​4bij​​,Jij​=4bij​​.

Please Run main.py
