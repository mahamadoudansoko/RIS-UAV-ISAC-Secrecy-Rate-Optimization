<<<<<<< HEAD
# DRL for RIS-Assisted UAV-ISAC Systems

This repository contains the Python implementation of a Deep Deterministic Policy Gradient (DDPG) agent for solving the joint trajectory and resource allocation problem in a RIS-assisted Integrated Sensing and Communication (ISAC) system with a jamming UAV for physical layer security.

## Project Structure

- `config.py`: Central configuration file for all system, channel, task, and DRL parameters.
- `environment.py`: Implements the custom OpenAI Gym environment for the UAV-ISAC system.
- `ddpg_agent.py`: Contains the DDPG agent, including Actor/Critic networks and Replay Buffer.
- `utils.py`: Contains utility functions, including exploration noise and plotting functions.
- `main.py`: The main script to execute the DRL training loop.
- `evaluate.py`: Script to load a trained model and generate performance plots.
- `requirements.txt`: A list of required Python packages.
- `/models/`: Directory where trained models will be saved.
- `/plots/`: Directory where generated plots will be saved.

## Setup

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Training the Agent

To start the training process, simply run the `main.py` script:

```bash
python main.py
=======
# UAV–RIS–ISAC System Optimization with DRL

<div align="center">
  <img width="800" alt="UAV RIS ISAC Diagram" src="https://github.com/user-attachments/assets/23f65a79-ac4c-4dbc-b435-a9fa19dd2e7b" />
</div>

<br/>

## 📌 Overview
This project investigates **joint trajectory and resource allocation optimization** in RIS-assisted UAV-enabled **ISAC** (Integrated Sensing and Communication) systems. 

It combines optimization-based and learning-based approaches to enhance:
* 📡 Communication efficiency
* 🎯 Sensing accuracy
* 🔐 System security

---

### 📚 References & Background
The work builds upon the following research:

1. **“Joint Trajectory and Resource Allocation Design for RIS-Assisted UAV-Enabled ISAC Systems”** by Wu et al.
2. **“Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning”** by Haoran Peng et al.

---

## 📝 Project Description

### 🔹 Stage 1: MATLAB Implementation (Reproduction)
We first reproduced the simulation results (Fig. 2, Fig. 3, and Fig. 4) from Wu et al.’s paper using MATLAB.

* **Optimization:** Performed via the **CVX** solver.
* **Key Features:** Implemented trajectory planning, power allocation, and sensing-communication trade-offs.

### 🔹 Stage 2: DRL-Based Security Enhancement
To extend the original work, we introduced security-aware optimization using **Deep Reinforcement Learning (DRL)**.

* **Algorithm:** Implemented a **Deep Deterministic Policy Gradient (DDPG)** algorithm in Python.
* **Codebase:** Adapted and extended code from the GitHub repository **UAV-RIS EnergyHarvesting** by Haoran Peng et al.
* **Focus:** Secure communication, trajectory adaptation, and robust RIS phase control under dynamic conditions.

---

## 🛠 Technologies & Tools

| Component | Technology / Tool |
| :--- | :--- |
| **Optimization** | **MATLAB / CVX** (Convex optimization for resource allocation) |
| **AI Model** | **Python / PyTorch** (DRL model training via DDPG) |
| **Data Viz** | **NumPy, Matplotlib** (Data manipulation and visualization) |
| **Simulation** | **Gym Environment** (Custom UAV–RIS–ISAC simulation setup) |
>>>>>>> d6ffb00989799e4f2f349d2d02657dad36132fe0
