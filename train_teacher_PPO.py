# train_teacher_PPO.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from glider_env import GliderEnv

# ✅ Глобальная переменная для отслеживания текущей сложности
current_difficulty = 1

def make_env(seed=0):
    def _init():
        env = GliderEnv(mode="teacher", randomize=True, difficulty=current_difficulty)
        env = Monitor(env)
        return env
    return _init

class CurriculumCallback(BaseCallback):
    """Автоматически повышает сложность, когда агент справляется с текущим уровнем."""
    
    def __init__(self, check_freq=50000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_threshold = 0.5  # 50% успешных полётов для перехода на следующий уровень
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            global current_difficulty
            
            # Проверяем успешность на текущем уровне
            # Упрощённо: смотрим на eval reward
            # Если eval reward > 100, значит агент часто долетает до цели (+200)
            # и можно повысить сложность
            
            # Получаем последнюю eval награду из logger
            if hasattr(self, 'logger') and self.logger is not None:
                try:
                    # Проверяем, есть ли успешные эпизоды
                    ep_rew = self.logger.name_to_value.get('eval/mean_reward', -1000)
                    
                    if current_difficulty == 1 and ep_rew > 50:
                        current_difficulty = 2
                        print(f"\n✅ ПЕРЕХОД НА УРОВЕНЬ 2 (средняя дистанция + препятствия)")
                    elif current_difficulty == 2 and ep_rew > 80:
                        current_difficulty = 3
                        print(f"\n✅ ПЕРЕХОД НА УРОВЕНЬ 3 (дальняя дистанция + ветер)")
                    elif current_difficulty == 3 and ep_rew > 100:
                        current_difficulty = 4
                        print(f"\n✅ ПЕРЕХОД НА УРОВЕНЬ 4 (полная рандомизация)")
                except Exception:
                    pass
        return True

def main():
    global current_difficulty
    n_envs = 8
    total_timesteps = 10_000_000  # Увеличим до 10 млн шагов
    
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999)])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,      # Умеренное исследование
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/teacher_curriculum/",
        policy_kwargs={
            "net_arch": dict(pi=[256, 128], vf=[256, 128])
        }
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/teacher_curriculum/best_model/",
        log_path="./logs/teacher_curriculum/eval/",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    curriculum_callback = CurriculumCallback(check_freq=50000)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./logs/teacher_curriculum/checkpoints/",
        name_prefix="teacher_curr"
    )
    
    print(f"🚀 Запуск curriculum learning. Начальный уровень: {current_difficulty}")
    print(f"Всего шагов: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, curriculum_callback, checkpoint_callback],
        reset_num_timesteps=True
    )
    
    model.save("./logs/teacher_curriculum/final_teacher_model")
    print(f"✅ Обучение завершено! Финальный уровень сложности: {current_difficulty}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()