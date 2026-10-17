# train_teacher.py
import os
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from glider_env import GliderEnv

def make_env(seed=0):
    def _init():
        env = GliderEnv(mode="teacher", randomize=True)
        env = Monitor(env)
        return env
    return _init

def main():
    # Гиперпараметры PPO (на основе рекомендаций для авиационных задач)
    n_envs = 8  # Количество параллельных сред
    total_timesteps = 5_000_000  # 5 млн шагов для начала
    
    # Создаём векторизованную среду
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999)])
    
    # Используем RecurrentPPO (PPO + LSTM) согласно ТЗ
    model = RecurrentPPO(
        "MlpLstmPolicy",  # MLP + LSTM для обработки последовательностей
        env,
        learning_rate=3e-4,
        n_steps=2048,       # Длина rollout
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # Поощрение исследования
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/teacher/",
        policy_kwargs={
            "lstm_hidden_size": 128,
            "enable_critic_lstm": True,
            "net_arch": dict(pi=[256, 128], vf=[256, 128])
        }
    )
    
    # Callback'и для сохранения и оценки
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/teacher/best_model/",
        log_path="./logs/teacher/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/teacher/checkpoints/",
        name_prefix="teacher_ppo"
    )
    
    print(f"Начинаем обучение учителя на {n_envs} параллельных средах...")
    print(f"Всего шагов: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=True
    )
    
    # Сохраняем финальную модель
    model.save("./logs/teacher/final_teacher_model")
    print("Обучение завершено! Модель сохранена.")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()