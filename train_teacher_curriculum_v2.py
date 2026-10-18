# train_teacher_curriculum_v6.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from glider_env import GliderEnv

current_difficulty = 1  
stable_checks_count = 0
REQUIRED_STABLE_CHECKS = 5  # Увеличили до 5 для большей стабильности

def make_env(seed=0):
    def _init():
        global current_difficulty
        env = GliderEnv(mode="teacher", randomize=True, difficulty=current_difficulty)
        env = Monitor(env)
        return env
    return _init

class CurriculumCallbackV6(BaseCallback):
    """Callback с защитой от коллапса: откат к лучшей модели при падении reward."""
    
    def __init__(self, train_env, eval_env, check_freq=100000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_check = 0
        self.best_reward = -float('inf')
        self.best_model_path = None
        
    def _update_difficulty_in_env(self, env, new_difficulty):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'difficulty'):
            base_env.difficulty = new_difficulty
    
    def _on_step(self) -> bool:
        global current_difficulty, stable_checks_count
        
        if self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps
            
            # Запускаем eval эпизоды
            goals_reached = 0
            episode_rewards = []
            
            for ep in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0
                goal_reached = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    if isinstance(done, (list, np.ndarray)):
                        done = bool(done[0])
                    if isinstance(info, list):
                        info = info[0]
                    
                    episode_reward += reward
                    
                    if info.get('goal_dist', float('inf')) < 30:
                        goal_reached = True
                
                episode_rewards.append(episode_reward)
                if goal_reached:
                    goals_reached += 1
            
            avg_reward = np.mean(episode_rewards)
            success_rate = goals_reached / self.n_eval_episodes
            
            print(f"\n{'='*60}")
            print(f"📊 Curriculum Check (шаг {self.num_timesteps}):")
            print(f"   Текущий уровень: {current_difficulty}")
            print(f"   Средняя награда: {avg_reward:.1f}")
            print(f"   ✅ Успешных полётов: {success_rate*100:.0f}% ({goals_reached}/{self.n_eval_episodes})")
            print(f"   Лучшая награда: {self.best_reward:.1f}")
            print(f"{'='*60}")
            
            # ✅ ЗАЩИТА ОТ КОЛЛАПСА: если reward упал, откатываемся
            if avg_reward < self.best_reward - 150:  # Упало больше чем на 50
                print(f"   ⚠️ ОБНАРУЖЕН КОЛЛАПС! Откат к лучшей модели...")
                if self.best_model_path and os.path.exists(self.best_model_path):
                    # Загружаем лучшую модель
                    self.model.set_parameters(self.best_model_path)
                    print(f"   ✅ Откат выполнен")
                stable_checks_count = 0
                return True
            
            # Обновляем лучшую модель
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.best_model_path = f"./logs/teacher_curriculum_v6/best_model_backup_{self.num_timesteps}.zip"
                self.model.save(self.best_model_path)
                print(f"   💾 Сохранена лучшая модель: {self.best_model_path}")
            
            # Проверка перехода на следующий уровень
            thresholds = {
                1: 1.0,
                2: 0.8,
                3: 0.6,  # 60% успешных полётов на уровне 3
            }
            
            if current_difficulty in thresholds:
                req_success = thresholds[current_difficulty]
                
                if success_rate >= req_success:
                    stable_checks_count += 1
                    print(f"   ✅ Стабильная работа на уровне {current_difficulty}")
                    
                    if stable_checks_count >= REQUIRED_STABLE_CHECKS:
                        new_difficulty = current_difficulty + 1
                        current_difficulty = new_difficulty
                        
                        if hasattr(self.eval_env, 'envs'):
                            for env in self.eval_env.envs:
                                self._update_difficulty_in_env(env, new_difficulty)
                        
                        self.train_env.set_attr('difficulty', new_difficulty)
                        stable_checks_count = 0
                        print(f"   🚀 ПЕРЕХОД НА УРОВЕНЬ {new_difficulty}")
                else:
                    stable_checks_count = 0
                    print(f"   ⏳ Агент ещё не стабилен на уровне {current_difficulty}")
            else:
                print(f"   🎯 Максимальный уровень достигнут")
        
        return True

def main():
    global current_difficulty, stable_checks_count
    
    n_envs = 8
    total_timesteps = 5_000_000  # Продолжаем ещё 5 млн шагов
    
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999)])
    
    # ✅ ЗАГРУЖАЕМ ЧЕКПОИНТ ДО КОЛЛАПСА
    checkpoint_path = "./logs/teacher_curriculum_v5/checkpoints/teacher_curr_v5_1300000_steps.zip"
    
    if os.path.exists(checkpoint_path):
        print(f"📂 Загрузка чекпоинта до коллапса: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env)
        # ✅ БОЛЕЕ КОНСЕРВАТИВНЫЕ ПАРАМЕТРЫ
        model.learning_rate = 1e-4  # Снижаем с 3e-4 до 1e-4
        model.ent_coef = 0.01       # Снижаем с 0.02 до 0.01
        model.clip_range = 0.15     # Снижаем с 0.2 до 0.15
    else:
        print(f"⚠️ Чекпоинт не найден: {checkpoint_path}")
        print("Создаём новую модель...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/teacher_curriculum_v6/",
            policy_kwargs={
                "net_arch": dict(pi=[256, 128], vf=[256, 128])
            }
        )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/teacher_curriculum_v6/best_model/",
        log_path="./logs/teacher_curriculum_v6/eval/",
        eval_freq=20000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    curriculum_callback = CurriculumCallbackV6(
        train_env=env,
        eval_env=eval_env,
        check_freq=100000,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Сохраняем чаще для быстрого отката
        save_path="./logs/teacher_curriculum_v6/checkpoints/",
        name_prefix="teacher_curr_v6"
    )
    
    print(f"🚀 Запуск curriculum learning v6 с защитой от коллапса")
    print(f"Начальный уровень: {current_difficulty}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Entropy coef: {model.ent_coef}")
    print(f"Clip range: {model.clip_range}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, curriculum_callback, checkpoint_callback],
        reset_num_timesteps=False
    )
    
    model.save("./logs/teacher_curriculum_v6/final_teacher_model")
    print(f"✅ Обучение завершено!")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
