import numpy as np
from gym_togyzkumalak.envs.togyzkumalak_env import TogyzkumalakEnv

class TogyzkumalakSelfPlayEnv(TogyzkumalakEnv):
    def __init__(self, opponent_model):
        super().__init__()
        self.opponent_model = opponent_model

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # Ход агента
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Ход противника
        obs = self._opponent_step(obs)

        return obs, reward, terminated, truncated, info

    def _opponent_step(self, obs):
        try:
            obs_opponent = obs.reshape((128, 1)).astype(np.float32)
            action, _ = self.opponent_model.predict(obs_opponent, deterministic=True)
            
            obs, _, _, _, _ = super().step(int(action))
        except Exception as e:
            # Если противник сделал невалидный ход — это критично, но логируем
            print(f"[!] Ошибка противника: {e}")
        return obs
