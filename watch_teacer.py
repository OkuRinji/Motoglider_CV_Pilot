# watch_teacher.py
import numpy as np
from stable_baselines3 import PPO
from glider_env import GliderEnv
import motoglider as mg

def main():
    # Загружаем лучшую модель
    model = PPO.load("./logs/teacher/best_model/best_model")
    
    # Создаём среду для визуализации
    env = GliderEnv(mode="teacher", randomize=False, difficulty=1)
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    for step in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: dist={info['goal_dist']:.1f}m, reward={reward:.2f}, total={total_reward:.2f}")
        
        if terminated or truncated:
            print(f"\nЭпизод завершен на шаге {step}")
            print(f"Причина: {info['reason'] if info['crashed'] else 'цель достигнута' if info['goal_dist'] < 30 else 'время вышло'}")
            print(f"Итоговая награда: {total_reward:.2f}")
            break
    
    env.close()

if __name__ == "__main__":
    main()