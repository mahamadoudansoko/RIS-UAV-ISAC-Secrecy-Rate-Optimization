# environment.py

import gym
from gym import spaces
import numpy as np
from config import SYSTEM_PARAMS, CHANNEL_PARAMS, TASK_PARAMS, REWARD_WEIGHTS

class UAV_ISAC_Environment(gym.Env):
    def __init__(self, config_override=None):
        super(UAV_ISAC_Environment, self).__init__()

        # Allow overriding config for evaluation purposes
        self.sp = SYSTEM_PARAMS
        self.cp = CHANNEL_PARAMS
        self.tp = TASK_PARAMS
        self.rw = REWARD_WEIGHTS
        if config_override:
            self.rw.update(config_override)

        # State space: [q_c(2), q_j(2), q_c_end(2), q_j_end(2), running_rates(K), running_snr(1)]
        self.state_dim = 2 + 2 + 2 + 2 + self.sp['K_USERS'] + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # Action space (normalized to [-1, 1]): [v_c(2), v_j(2), p_c(1), p_j(1), ris_phases(M)]
        self.action_dim = 2 + 2 + 1 + 1 + self.sp['M_RIS_ELEMENTS']
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

    def reset(self):
        self.q_c = self.sp['UAV_C_START'].copy().astype(float)
        self.q_j = self.sp['UAV_J_START'].copy().astype(float)
        
        self.running_avg_rates = np.zeros(self.sp['K_USERS'])
        self.running_avg_snr = 0.0
        self.current_step = 0
        
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        
        # 1. Parse and scale action from [-1, 1] to physical values
        v_c = action[0:2] * self.sp['V_MAX']
        v_j = action[2:4] * self.sp['V_MAX']
        p_c = (action[4] + 1) / 2 * self.sp['P_C_MAX_W']
        p_j = (action[5] + 1) / 2 * self.sp['P_J_MAX_W']
        
        # 2. Update UAV positions
        self.q_c += v_c * self.tp['DT_S']
        self.q_j += v_j * self.tp['DT_S']

        # 3. Calculate metrics (Simplified Placeholder)
        # ** THIS IS A KEY SIMPLIFICATION **
        # Replace this with the full channel model and SINR/SNR equations from the paper.
        comm_rate, sensing_snr, secrecy_rate = self._calculate_metrics(p_c, p_j)

        # 4. Calculate reward
        reward = self._calculate_reward(comm_rate, sensing_snr, p_c, p_j)

        # 5. Update running averages for QoS checks
        self.running_avg_rates = (self.running_avg_rates * (self.current_step - 1) + comm_rate) / self.current_step
        self.running_avg_snr = (self.running_avg_snr * (self.current_step - 1) + sensing_snr) / self.current_step
        
        # 6. Get next state and done flag
        next_state = self._get_state()
        done = (self.current_step >= self.tp['N_STEPS'])
        
        # Store metrics for logging/evaluation
        info = {'comm_rate': comm_rate, 'sensing_snr': sensing_snr, 'secrecy_rate': secrecy_rate}

        return next_state, reward, done, info

    def _get_state(self):
        return np.concatenate([
            self.q_c, self.q_j,
            self.sp['UAV_C_END'], self.sp['UAV_J_END'],
            self.running_avg_rates,
            [self.running_avg_snr]
        ]).astype(np.float32)

    def _calculate_path_loss(self, p1, p2, h1, h2, alpha):
        dist_3d = np.sqrt(np.sum((p1 - p2)**2) + (h1 - h2)**2)
        return self.cp['BETA_0'] * (dist_3d**(-alpha))

    def _calculate_metrics(self, p_c, p_j):
        """
        **PLACEHOLDER METRIC CALCULATION**
        This function is a simplified model. A full implementation would involve:
        1. Calculating all channel matrices (h_Ck, h_CR, g_Rk, h_CS, etc.).
        2. Constructing the effective channels (H_eff_Ck, H_eff_CS).
        3. Using the DFRC beamformers from the action to calculate SINR and SNR.
        4. Calculating secrecy rate from SINR at CU and Eve.
        """
        comm_rates = np.zeros(self.sp['K_USERS'])
        for k in range(self.sp['K_USERS']):
            # Simplified: Signal strength depends on distance
            user_loc = self.sp['USER_LOCATIONS'][k]
            pl_c_k = self._calculate_path_loss(self.q_c, user_loc, self.sp['H_UAV'], 0, self.cp['ALPHA_LOS'])
            pl_j_k = self._calculate_path_loss(self.q_j, user_loc, self.sp['H_UAV'], 0, self.cp['ALPHA_LOS'])
            
            signal = p_c * pl_c_k
            interference = p_j * pl_j_k + self.cp['NOISE_POWER_W']
            sinr = signal / interference
            comm_rates[k] = np.log2(1 + sinr)

        # Simplified sensing SNR
        pl_c_s = self._calculate_path_loss(self.q_c, self.sp['TARGET_LOCATION'], self.sp['H_UAV'], 0, self.cp['ALPHA_LOS'])
        sensing_snr = (p_c * pl_c_s) / self.cp['NOISE_POWER_W']
        
        # Simplified secrecy rate (assuming Eve's SINR is some fraction of the first user's)
        pl_c_e = self._calculate_path_loss(self.q_c, self.sp['EVE_LOCATION_ESTIMATE'], self.sp['H_UAV'], 0, self.cp['ALPHA_LOS'])
        sinr_eve = (p_c * pl_c_e) / self.cp['NOISE_POWER_W']
        secrecy_rate = max(0, comm_rates[0] - np.log2(1+sinr_eve))

        return comm_rates, sensing_snr, secrecy_rate

    def _calculate_reward(self, comm_rate, sensing_snr, p_c, p_j):
        # Weighted objective from the paper (trade-off)
        beta_c = self.rw['BETA_C']
        weighted_reward = beta_c * np.sum(comm_rate) + (1 - beta_c) * np.log(1 + sensing_snr) # log for scale
        
        # Penalty for low sensing performance
        sense_penalty = self.rw['LAMBDA_S'] * max(0, self.tp['GAMMA_S_MIN'] - sensing_snr)
        
        # Penalty for low communication QoS
        qos_penalty = self.rw['LAMBDA_Q'] * np.sum([max(0, self.tp['R_K_MIN'] - r) for r in self.running_avg_rates])
        
        # Penalty for power consumption
        power_penalty = self.rw['LAMBDA_P'] * (p_c / self.sp['P_C_MAX_W'] + p_j / self.sp['P_J_MAX_W'])

        # Penalty for collision
        dist_uavs = np.linalg.norm(self.q_c - self.q_j)
        collision_penalty = self.rw['LAMBDA_C'] if dist_uavs < self.rw['D_MIN'] else 0.0
        
        total_reward = weighted_reward - sense_penalty - qos_penalty - power_penalty - collision_penalty
        return total_reward