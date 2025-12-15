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