# In file: UAV-RIS_EnergyHarvesting/gym_foo/foo_env.py

import gym
from gym import spaces
import numpy as np

class SecrecyISACEnv(gym.Env):
    """
    Custom Gym environment for a MULTI-USER UAV-RIS ISAC system.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, w_S=0.5, Train=True, multiUT=True, LoadData=False, 
                 Trajectory_mode='Kmeans', MaxStep=50, 
                 NUM_RIS_ELEMENTS=100, P_ISAC_MAX=10**((36 - 30) / 10)):
        
        super(SecrecyISACEnv, self).__init__()

        # =========================================================================
        # 1. DEFINE SYSTEM PARAMETERS
        # =========================================================================
        self.NUM_CU = 2
        self.NUM_RIS_ELEMENTS = NUM_RIS_ELEMENTS
        
        self.ISAC_UAV_ALTITUDE = 100.0
        self.JAMMER_UAV_ALTITUDE = 100.0
        self.RIS_ALTITUDE = 15.0
        
        self.loc_ris = np.array([150.0, 90.0, self.RIS_ALTITUDE])
        self.loc_cu = np.array([[230.0, 250.0, 0.0], [250.0, 230.0, 0.0]])
        self.loc_target = np.array([60.0, 235.0, 0.0])
        self.loc_eve = np.array([200.0, 100.0, 0.0])

        self.P_ISAC_MAX = P_ISAC_MAX
        self.P_JAMMER_MAX = self.P_ISAC_MAX
        
        self.NOISE_POWER_W = 10**(-114 / 10)
        self.PATH_LOSS_EXPONENT_LOS = 2.2
        self.RICIAN_FACTOR = 10
        self.BETA_0 = 10**(-30 / 10)

        self.FIELD_DIM = 300.0
        self.MAX_STEPS = int(MaxStep)
        self.TIME_SLOT_DURATION = 1.0
        self.V_MAX = 20.0
        self.w_S = w_S
        self.GAMMA_S_TARGET = 10**(5 / 10)
        self.penalty_boundary = 10
        
        # =========================================================================
        # 2. DEFINE STATE AND ACTION SPACES
        # =========================================================================
        state_dim = 2 + 2 + (2 * self.NUM_CU) + 1
        self.observation_space = spaces.Box(low=0, high=self.FIELD_DIM, shape=(state_dim,), dtype=np.float32)
        
        action_dim = 2 + 2 + 1 + 1 + self.NUM_RIS_ELEMENTS
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        
        # This attribute is needed by some Gym wrappers
        self._max_episode_steps = self.MAX_STEPS

        # Initialize the state variables immediately
        self.reset()

    def reset(self):
        self.isac_uav_pos = np.array([0.0, 150.0, self.ISAC_UAV_ALTITUDE])
        self.jammer_uav_pos = np.array([0.0, 150.0, self.JAMMER_UAV_ALTITUDE])
        self.ris_phases = np.random.uniform(-np.pi, np.pi, self.NUM_RIS_ELEMENTS)
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        cu_positions_flat = self.loc_cu[:, :2].flatten() / self.FIELD_DIM
        state = np.concatenate([
            self.isac_uav_pos[:2] / self.FIELD_DIM,
            self.jammer_uav_pos[:2] / self.FIELD_DIM,
            cu_positions_flat,
            [(self.MAX_STEPS - self.current_step) / self.MAX_STEPS]
        ])
        return state.astype(np.float32)

    def _calculate_channel_gain(self, pos1, pos2):
        dist = np.linalg.norm(pos1 - pos2)
        dist = max(dist, 1e-6)
        path_loss = self.BETA_0 * (dist ** -self.PATH_LOSS_EXPONENT_LOS)
        return path_loss
    
    def _calculate_rician_channel_vec(self, pos_ris, pos_rx):
        dist_ris_rx = np.linalg.norm(pos_ris - pos_rx)
        h_los_gain = self._calculate_channel_gain(pos_ris, pos_rx)
        h_nlos_gain = h_los_gain
        h_los = np.sqrt(h_los_gain)
        h_nlos = np.sqrt(h_nlos_gain) * (np.random.randn(self.NUM_RIS_ELEMENTS) + 1j * np.random.randn(self.NUM_RIS_ELEMENTS)) / np.sqrt(2)
        K = self.RICIAN_FACTOR
        return np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos

    def step(self, action):
        self.current_step += 1
        isac_v = action[0:2] * self.V_MAX
        jammer_v = action[2:4] * self.V_MAX
        p_isac = ((action[4] + 1) / 2) * self.P_ISAC_MAX
        p_jammer = ((action[5] + 1) / 2) * self.P_JAMMER_MAX
        self.ris_phases = action[6:] * np.pi
        ris_phase_matrix = np.diag(np.exp(1j * self.ris_phases))
        self.isac_uav_pos[:2] += isac_v * self.TIME_SLOT_DURATION
        self.jammer_uav_pos[:2] += jammer_v * self.TIME_SLOT_DURATION
        reward_penalty = 0
        for pos in [self.isac_uav_pos, self.jammer_uav_pos]:
            if not (0 <= pos[0] <= self.FIELD_DIM and 0 <= pos[1] <= self.FIELD_DIM):
                reward_penalty -= self.penalty_boundary
                pos[0] = np.clip(pos[0], 0, self.FIELD_DIM)
                pos[1] = np.clip(pos[1], 0, self.FIELD_DIM)
        h_ae = np.sqrt(self._calculate_channel_gain(self.isac_uav_pos, self.loc_eve))
        h_je = np.sqrt(self._calculate_channel_gain(self.jammer_uav_pos, self.loc_eve))
        h_ar_gain = self._calculate_channel_gain(self.isac_uav_pos, self.loc_ris)
        h_ar_vec = np.full(self.NUM_RIS_ELEMENTS, np.sqrt(h_ar_gain))
        h_re_vec = self._calculate_rician_channel_vec(self.loc_ris, self.loc_eve)
        h_eff_e = h_ae + h_re_vec @ ris_phase_matrix @ h_ar_vec
        signal_e = p_isac * (np.abs(h_eff_e)**2)
        interference_e = p_jammer * (np.abs(h_je)**2)
        sinr_e = signal_e / (interference_e + self.NOISE_POWER_W)
        rate_e = np.log2(1 + sinr_e)
        potential_secrecy_rates = []
        for k in range(self.NUM_CU):
            cu_loc = self.loc_cu[k]
            h_ac_k = np.sqrt(self._calculate_channel_gain(self.isac_uav_pos, cu_loc))
            h_jc_k = np.sqrt(self._calculate_channel_gain(self.jammer_uav_pos, cu_loc))
            h_rc_k_vec = self._calculate_rician_channel_vec(self.loc_ris, cu_loc)
            h_eff_c_k = h_ac_k + h_rc_k_vec @ ris_phase_matrix @ h_ar_vec
            signal_c_k = p_isac * (np.abs(h_eff_c_k)**2)
            interference_c_k = p_jammer * (np.abs(h_jc_k)**2)
            sinr_c_k = signal_c_k / (interference_c_k + self.NOISE_POWER_W)
            rate_c_k = np.log2(1 + sinr_c_k)
            secrecy_rate_k = max(0, rate_c_k - rate_e)
            potential_secrecy_rates.append(secrecy_rate_k)
        scheduled_user_idx = np.argmax(potential_secrecy_rates)
        achieved_secrecy_rate = potential_secrecy_rates[scheduled_user_idx]
        h_as = np.sqrt(self._calculate_channel_gain(self.isac_uav_pos, self.loc_target))
        h_rs_vec = self._calculate_rician_channel_vec(self.loc_ris, self.loc_target)
        h_eff_s = h_as + h_rs_vec @ ris_phase_matrix @ h_ar_vec
        sensing_signal_power = p_isac * (np.abs(h_eff_s)**2)
        sensing_snr = sensing_signal_power / self.NOISE_POWER_W
        reward_sensing = self.w_S * max(0, sensing_snr - self.GAMMA_S_TARGET)
        reward = achieved_secrecy_rate + reward_sensing + reward_penalty
        done = self.current_step >= self.MAX_STEPS
        info = {
            'secrecy_rate': achieved_secrecy_rate,
            'sensing_snr_db': 10 * np.log10(sensing_snr) if sensing_snr > 0 else -100,
            'scheduled_user': scheduled_user_idx,
            'isac_pos': self.isac_uav_pos[:2],
            'jammer_pos': self.jammer_uav_pos[:2],
        }
        return self._get_state(), reward, done, info

    def close(self):
        pass