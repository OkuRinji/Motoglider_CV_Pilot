# train_teacher_curriculum_v2.py
"""
Обучение сети-учителя с curriculum learning по сложности.
Награды строго по ТЗ (Таблица 6).
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from glider_env import GliderEnv

current_difficulty = 1
stable_checks_count = 0
REQUIRED_STABLE_CHECKS = 3


def make_env(seed=0):
    def _init():
        global current_difficulty
        env = GliderEnv(mode="teacher", randomize=True, difficulty=current_difficulty)
        env = Monitor(env)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Автоматический переход на следующий уровень сложности при стабильном успехе."""

    def __init__(self, train_env, eval_env, check_freq=100000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_check = 0

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
            print(f"   Уровень сложности: {current_difficulty}")
            print(f"   Средняя награда: {avg_reward:.2f}")
            print(f"   ✅ Успешных полётов: {success_rate*100:.0f}% ({goals_reached}/{self.n_eval_episodes})")
            print(f"{'='*60}")

            # Пороги успешности для перехода на следующий уровень
            thresholds = {1: 0.8, 2: 0.7, 3: 0.6}

            if current_difficulty in thresholds:
                req_success = thresholds[current_difficulty]

                if success_rate >= req_success:
                    stable_checks_count += 1
                    print(f"   ✅ Стабильная работа (проверка {stable_checks_count}/{REQUIRED_STABLE_CHECKS})")

                    if stable_checks_count >= REQUIRED_STABLE_CHECKS:
                        new_difficulty = current_difficulty + 1
                        current_difficulty = new_difficulty

                        # Обновляем difficulty во всех средах через set_attr
                        self.train_env.set_attr('difficulty', new_difficulty)
                        self.eval_env.set_attr('difficulty', new_difficulty)

                        stable_checks_count = 0
                        print(f"   🚀 ПЕРЕХОД НА УРОВЕНЬ СЛОЖНОСТИ {new_difficulty}")
                else:
                    stable_checks_count = 0
                    print(f"   ⏳ Нужно: успех > {req_success*100}%")
            else:
                print(f"   🎯 Достигнут максимальный уровень сложности (4)")

        return True


def main():
    global current_difficulty, stable_checks_count

    n_envs = 36  # Оптимально для Xeon 1650 v3 (12 потоков × 2)
    total_timesteps = 5_000_000

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999)])

    LOG_DIR = "./logs/teacher_curriculum_v8/"

    # Загружаем существующую модель, если есть
    model_path = "./logs/teacher_curriculum_v7/best_model/best_model.zip"

    if os.path.exists(model_path):
        print(f"📂 Загрузка модели: {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
        model.learning_rate = 3e-4
        model.ent_coef = 0.02
        model.tensorboard_log = LOG_DIR
    else:
        print("🆕 Создание новой модели")
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
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device="cpu",
            tensorboard_log=LOG_DIR,
            policy_kwargs={
                "net_arch": dict(pi=[256, 128], vf=[256, 128])
            }
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model/"),
        log_path=os.path.join(LOG_DIR, "eval/"),
        eval_freq=20000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    curriculum_callback = CurriculumCallback(
        train_env=env,
        eval_env=eval_env,
        check_freq=100000,
        n_eval_episodes=10
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=os.path.join(LOG_DIR, "checkpoints/"),
        name_prefix="teacher_curr_v8"
    )

    print(f"🚀 Запуск обучения с наградами по ТЗ")
    print(f"   Начальный уровень сложности: {current_difficulty}")
    print(f"   n_envs: {n_envs}")
    print(f"   Всего шагов: {total_timesteps}")
    print(f"   Логи: {LOG_DIR}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, curriculum_callback, checkpoint_callback],
        reset_num_timesteps=False
    )

    model.save(os.path.join(LOG_DIR, "final_teacher_model"))
    print(f"✅ Обучение завершено!")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()