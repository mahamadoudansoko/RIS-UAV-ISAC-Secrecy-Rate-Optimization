
**UAV–RIS–ISAC System Optimization with DRL**

<img width="726" height="534" alt="image" src="https://github.com/user-attachments/assets/23f65a79-ac4c-4dbc-b435-a9fa19dd2e7b" />

**OVERVIEW**
This project investigates joint trajectory and resource allocation optimization in RIS-assisted UAV-enabled ISAC (Integrated Sensing and Communication) systems.
It combines optimization-based and learning-based approaches to enhance communication efficiency, sensing accuracy, and system security.


*The work builds upon:**

1.) “Joint Trajectory and Resource Allocation Design for RIS-Assisted UAV-Enabled ISAC Systems” by Wu et al.

2.) “Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning” by Haoran Peng et al.

**PROJECT DESCRIPTION
Stage 1 **MATLAB Implementation

We first reproduced the simulation results (Fig. 2, Fig. 3, and Fig. 4) from Wu et al.’s paper using MATLAB.

a.) Optimization was performed via the **CVX** solver.

b.) Implemented trajectory planning, power allocation, and sensing-communication trade-offs.

Stage 2  **DRL-Based Security** Enhancement

To extend the original work, we introduced security-aware optimization using Deep Reinforcement Learning (DRL):

Implemented a Deep Deterministic Policy Gradient (DDPG) algorithm in Python.

Adapted and extended code from the GitHub repository **UAV-RIS EnergyHarvesting
 by Haoran Peng et al.

Focused on secure communication, trajectory adaptation, and robust RIS phase control under dynamic conditions.

**Technologies & Tools**

   MATLAB / CVX – Convex optimization for resource allocation.
    
   Python / PyTorch – DRL model training (DDPG).
    
   NumPy, Matplotlib – Data manipulation and visualization.
    
   Gym Environment – Custom UAV–RIS–ISAC simulation setup.

