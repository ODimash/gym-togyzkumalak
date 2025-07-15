# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv

# from gym_togyzkumalak.envs.TogyzkumalakSelfPlayEnv import TogyzkumalakSelfPlayEnv


# model = PPO.load("togyzkumalak_ppo", verbose=1)
# old_model = PPO.load("togyzkumalak_ppo")

# # Оборачиваем новую среду с self-play
# from stable_baselines3.common.vec_env import DummyVecEnv

# def make_env():
#     env = TogyzkumalakSelfPlayEnv(opponent_model=old_model)
#     return Monitor(env)

# env = DummyVecEnv([make_env])

# # Устанавливаем новую среду в загруженную модель
# model.set_env(env)

# # Продолжаем обучение
# model.learn(total_timesteps=500_000)
# model.save("togyzkumalak_ppo_v2")


# Первая тренировка модели
#
import gymnasium as gym
import gym_togyzkumalak
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from gym_togyzkumalak.envs.togyzkumalak_env import TogyzkumalakEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    def _init():
        env = TogyzkumalakEnv()
        return Monitor(env)
    return _init

def test_trained_model(model):
    test_env = TogyzkumalakEnv()
    obs, _ = test_env.reset()
    for _ in range(1000):
        test_env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated
        print(f"Ход: {action}, Награда: {reward}")
        if done:
            print("Игра завершена!")
            break
    test_env.close()

if __name__ == '__main__':
    torch.set_num_threads(4)
    N_ENVS = 4
    TOTAL_TIMESTEPS = 90_000_000

    # Проверим среду на корректность
    check_env(make_env()())

    # Параллельные среды
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    # Попытка загрузить старую модель
    try:
        model = PPO.load("togyzkumalak_ppo", env=env, verbose=1)
        print("Модель успешно загружена.")
    except FileNotFoundError:
        print("Модель не найдена, создаётся новая.")
        model = PPO("MlpPolicy", env, verbose=1)

    # Обучение
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("togyzkumalak_ppo_v2")
    print("Модель сохранена.")

    # Тестирование
    test_trained_model(model)


# Игра с ботом
#
# from stable_baselines3 import PPO
# import gymnasium as gym
# import gym_togyzkumalak

# # Загрузка модели
# env = gym.make("Togyzkumalak-v0")
# model = PPO.load("togyzkumalak_ppo", env=env)

# obs, _ = env.reset()
# env.render()

# for turn in range(100):  # максимум 100 ходов
#     # === Ход человека ===
#     while True:
#         try:
#             action = int(input("Твой ход (0-8): "))
#             obs_copy = obs.copy()
#             obs_copy.flags.writeable = False  # на всякий случай

#             # проверим легальность хода
#             # if action not in env.action_space():
#             #     print("Нелегальный ход. Попробуй снова.")
#             #     continue
#             break
#         except ValueError:
#             print("Введи число от 0 до 8")

#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()

#     if terminated or truncated:
#         print("Игра завершена!")
#         break

#     # === Ход бота ===
#     action, _ = model.predict(obs, deterministic=True)
#     print(f"Ход бота: {action}")
#     obs, reward, terminated, truncated, info = env.step(int(action))
#     env.render()

#     if terminated or truncated:
#         print("Игра завершена!")
#         break
