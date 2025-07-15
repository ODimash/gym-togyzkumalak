import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from gym_togyzkumalak.envs.togyzkumalak_discrete import TogyzkumalakDiscrete
from gym_togyzkumalak.togyzkumalak.board import Board


class TogyzkumalakEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.__version__ = "0.0.2"

        self.board = Board()
        self.action_space = TogyzkumalakDiscrete(9, self.board)

        # Устанавливаем реальные границы наблюдений
        low = np.zeros((128, 1), dtype=np.float32)
        high = np.ones((128, 1), dtype=np.float32)

        # Устанавливаем поля, где реально может быть значение до 18
        for i in range(5, 56, 7):
            high[i, 0] = 18.0
        high[61, 0] = 18.0

        for i in range(63, 117, 7):
            high[i, 0] = 18.0
        high[123, 0] = 18.0

        high[60, 0] = 2.0

        # Также установим возможные значения до 2.0 для полей, где у тебя 1.1111
        for i in range(68, 111, 7):
            high[i, 0] = 2.0

        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        action = int(action)  
        # if (self.board.run.name == 'white'):
        #     action = input(f"Ход игрока {self.board.run.name} (1-9): ")
        try:
            observation, reward, done, info = self.board.move(action=action)
            terminated = done
        except Exception as e:
            # print(f"Нелегальный ход: {e}")
            # Нелегальный ход, заканчиваем эпизод с негативной наградой
            observation = self.board.observation()
            reward = -1.0
            terminated = True
            info = {"error": str(e)}

        self.action_space.update_board(self.board)
        truncated = False  # или True, если есть лимит по ходам
        return observation.astype(np.float32).reshape((128, 1)), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board()
        self.action_space.update_board(self.board)

        observation = self.board.observation().astype(np.float32).reshape((128, 1))
        return observation, {}

    def render(self):
        self.board.print()

    def check_action(self, action):
        return self.board.run.check_action(action)

    def observation(self):
        return self.board.observation()

    def available_action(self):
        return self.board.run.available_action()

    def reward(self):
        return self.board.reward
