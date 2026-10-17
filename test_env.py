# test_env.py
import numpy as np
from glider_env import GliderEnv

def test_env():
    print("Тестирование среды для СЕТИ-УЧЕНИКА...")
    env = GliderEnv(mode="student", randomize=True)
    
    obs, info = env.reset(seed=42)
    print(f"Форма depth: {obs['depth'].shape}")
    print(f"Форма vector: {obs['vector'].shape}")
    print(f"Начальная информация: {info}")
    
    total_reward = 0.0
    for i in range(50):  # 50 шагов = 5 секунд
        # Случайное действие для теста
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Эпизод завершен на шаге {i}. Причина: {info['reason']}")
            break
            
    print(f"Суммарная награда за тест: {total_reward:.4f}")
    env.close()
    print("Тест среды ученика пройден успешно!\n")

    print("Тестирование среды для СЕТИ-УЧИТЕЛЯ...")
    env_teacher = GliderEnv(mode="teacher", randomize=False)
    obs_t, info_t = env_teacher.reset(seed=42)
    print(f"Форма наблюдения учителя: {obs_t.shape}")
    
    action_t = env_teacher.action_space.sample()
    obs_t, reward_t, term_t, trunc_t, info_t = env_teacher.step(action_t)
    print(f"Награда на первом шаге: {reward_t:.4f}")
    env_teacher.close()
    print("Тест среды учителя пройден успешно!")

if __name__ == "__main__":
    test_env()