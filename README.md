<div align="center">

# 🛰️ RIS-Aided UAV-ISAC Secrecy Rate Optimization
### Secure & Resilient Integrated Sensing and Communication Networks

<!-- HEADER IMAGE -->
<img width="800" alt="UAV RIS ISAC Diagram" src="https://github.com/user-attachments/assets/23f65a79-ac4c-4dbc-b435-a9fa19dd2e7b" />

<br/>
<br/>

<!-- TECH STACK BADGES -->
<!-- Languages -->
<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/MATLAB-Optimization-e67e22?style=for-the-badge&logo=mathworks&logoColor=white" />

<!-- AI & ML -->
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Gymnasium-Farama-black?style=for-the-badge&logo=openai&logoColor=white" />
<img src="https://img.shields.io/badge/TensorBoard-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />

<!-- Data Science & Env -->
<img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-Data_Viz-11557c?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Miniconda-Env_Manager-44A833?style=for-the-badge&logo=anaconda&logoColor=white" />

</div>

---

## 📖 Abstract

The growing demand for secure and resilient **Integrated Sensing and Communication (ISAC)** networks has attracted significant attention, particularly in adversarial environments affected by jamming and eavesdropping threats. 

In this paper, we introduce a novel **Reconfigurable Intelligent Surface (RIS)-aided Unmanned Aerial Vehicle (UAV)-enabled ISAC framework**, designed to ensure the confidentiality and reliability of simultaneous Communication and Sensing (C&S) operations under strict secrecy constraints.

### 🎯 Objective
Our primary goal is to **maximize the secrecy rate** by jointly optimizing critical parameters:
1.  🚁 **UAV's Dynamic Trajectory**
2.  📡 **RIS Phase Shift Configuration**
3.  ⚡ **Transmit Power Allocation**
4.  👥 **User Scheduling**

To address this high-dimensional optimization problem, we propose an **AI-driven solution** based on the **Deep Deterministic Policy Gradient (DDPG)** algorithm—a reinforcement learning technique adapted for continuous control in dynamic environments.

> **Key Results:** The simulation demonstrates the superiority of our framework over benchmark schemes in terms of secrecy rate and overall system resilience against jamming.

**Index Terms:** _ISAC, UAV, RIS, Secrecy Rate Maximization, DDPG, Trajectory Optimization, Anti-Jamming._

---

## 🏗️ Project Architecture

This repository is divided into two distinct implementation stages:

### 🔹 Stage 1: MATLAB (Optimization Benchmark)
We reproduced simulation results from *Wu et al.* using the **CVX** solver to establish a baseline.
*   **Focus:** Trajectory planning, power allocation, and sensing-communication trade-offs.
*   **Location:** `/MATLAB` directory.

### 🔹 Stage 2: Python (DRL Security Enhancement)
We extended the work using **Deep Reinforcement Learning** to handle dynamic security threats.
*   **Algorithm:** DDPG (Actor-Critic Network).
*   **Environment:** Custom Gymnasium environment for UAV dynamics.
*   **Location:** `/PYTHON` directory.

---

## 📂 Repository Structure

```text
📦 RIS-UAV-ISAC-Secrecy-Rate-Optimization
 ┣ 📂 MATLAB                    # CVX Optimization Scripts (Stage 1)
 ┃ ┗ 📜 main_optimization.m     # (Add your MATLAB files here)
 ┃
 ┣ 📂 PYTHON                    # DRL Implementation Files (Stage 2)
 ┃ ┣ 📂 models                  # Saved DDPG Actor/Critic weights
 ┃ ┣ 📂 plots                   # Generated performance graphs
 ┃ ┣ 📜 config.py               # ⚙️ System, Channel & DRL Hyperparameters
 ┃ ┣ 📜 environment.py          # 🌍 Custom Gym Environment (UAV-ISAC Physics)
 ┃ ┣ 📜 ddpg_agent.py           # 🧠 The AI Agent (Actor-Critic + Replay Buffer)
 ┃ ┣ 📜 main.py                 # 🚀 Main Training Loop
 ┃ ┣ 📜 evaluate.py             # 📊 Testing & Plotting Script
 ┃ ┣ 📜 utils.py                # 🛠️ Helper functions (Noise, Math)
 ┃ ┗ 📜 requirements.txt        # 📦 Python Dependencies
 ┃
 ┣ 📂 Presentation              # Slides and visual assets
 ┣ 📂 Report                    # Thesis/Paper documentation
 ┗ 📜 README.md                 # 📄 Documentation