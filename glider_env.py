# glider_env.py
import gymnasium as gym
import numpy as np
import math
import motoglider as mg

DEG = math.pi / 180.0

class GliderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, mode="student", render_mode=None, randomize=True):
        super().__init__()
        self.mode = mode  # "teacher" или "student"
        self.render_mode = render_mode
        self.randomize = randomize
        
        # Инициализация симулятора
        depth_enabled = (mode == "student")
        self.sim = mg.GliderSim(depth=depth_enabled, depth_size=(64, 40))
        
        # --- Пространство действий ---
        # [крен (рад), скорость (м/с), скольжение (рад), газ (0..1), экстренный режим (0..1)]
        self.action_space = gym.spaces.Box(
            low=np.array([-45*DEG, 9.5, -10*DEG, 0.0, 0.0], dtype=np.float32),
            high=np.array([45*DEG, 20.0, 10*DEG, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # --- Пространство наблюдений ---
        if self.mode == "teacher":
            # 75 лучей + 3 (ветер) + 10 (навигация: V, alpha, beta, gamma, theta, omega_x,y,z, H, battery) 
            # + 3 (цель: dist, bearing, alt_diff) + N (препятствия: упрощенно возьмем фиксированный вектор)
            # Для простоты и стабильности обучениия объединим в 1D вектор
            obs_dim = 75 + 3 + 10 + 3 + 20  # 111
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        else:
            # Ученик: Карта глубины (64, 40) + ветер (3) + навигация (10) + цель (3)
            self.observation_space = gym.spaces.Dict({
                "depth": gym.spaces.Box(low=0.0, high=400.0, shape=(64, 40), dtype=np.float32),
                "vector": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
            })

        self.prev_action = np.zeros(5, dtype=np.float32)
        self.prev_dist = np.inf
        self.prev_motor_on = False
        self.episode_steps = 0
        self.max_episode_steps = 1500  # 150 секунд

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        wind_speed = np.random.uniform(0.0, 7.0)
        wind_dir = np.random.uniform(0.0, 360.0)
        
        if self.randomize:
            start, goal, heading = self.sim.randomize_scene(rng=np.random.default_rng(seed))
            s = self.sim.reset(start=start, goal=goal, heading_deg=heading, wind=(wind_speed, wind_dir))
        else:
            s = self.sim.reset(wind=(wind_speed, wind_dir))

        self.prev_action = np.array([0.0, 12.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_dist = s["goal_dist"]
        self.prev_motor_on = s["motor_on"]
        self.episode_steps = 0
        
        return self._get_obs(s), self._get_info(s)

    def step(self, action):
        self.episode_steps += 1
        
        # 1. Применяем действие
        gamma, v_ref, beta, throttle, emergency_flag = action
        elevator = None
        
        # Если флаг экстренного режима > 0.5, переходим в прямое управление РВ
        if emergency_flag > 0.5:
            # Для простоты, в экстренном режиме сеть может выдавать угол РВ в 5-м элементе action, 
            # но по ТЗ уставки: крен, скорость, скольжение, газ. 
            # Давайте интерпретировать 5-й элемент как переключатель, а угол РВ возьмем из отдельного выхода или масштабируем.
            # По ТЗ: "в экстренном режиме вместо скорости задается отклонение руля высоты".
            # Значит, если emergency > 0.5, мы используем v_ref как угол РВ (преобразуя диапазон 9.5..20 в -15..15 град)
            elevator = ((v_ref - 9.5) / (20.0 - 9.5)) * 30.0 * DEG - 15.0 * DEG
            v_ref = 12.0 # Фиктивная скорость, так как контур скорости отключен

        s = self.sim.step(gamma, v_ref, beta, throttle, elevator=elevator)

        # 2. Вычисляем награду (Таблица 6, пересчет на шаг 0.1 с)
        reward = 0.0
        
        # Приближение к цели: +0.01 за метр
        dist_change = self.prev_dist - s["goal_dist"]
        reward += 0.01 * dist_change
        
        # Достижение цели: +10
        if s["goal_dist"] < 30.0:
            reward += 10.0
            
        # Столкновение/разрушение/сваливание: -10
        if s["crashed"]:
            reward -= 10.0
            
        # Энергия мотора: -0.05 за кДж (за шаг)
        # sim.step не возвращает дельту энергии напрямую, но мы можем оценить по мощности или взять из состояния
        # В motoglider.py: self.E_motor += self.N * self.dt. Но в state() возвращается только общее E_motor.
        # Приблизительно: если мотор включен, штраф за шаг. Точнее: используем разницу E_motor, если бы она была, 
        # но проще штрафовать за сам факт работы мотора пропорционально throttle.
        # По ТЗ: "-0.05 за кДж". Возьмем оценку: N (мощность) * 0.1с = кДж. 
        # В state() есть 'power' (N, кВт? Нет, в motoglider.py N - это мощность в Вт? Проверим: self.N = P * (V + vi) / eta. Это Вт).
        # Значит, энергия за шаг = s["power"] * 0.1 / 1000.0 (кДж).
        energy_kj = s["power"] * 0.1 / 1000.0
        reward -= 0.05 * energy_kj
        
        # Включение мотора: -0.2 за каждое включение
        if s["motor_on"] and not self.prev_motor_on:
            reward -= 0.2
            
        # Высота меньше 30 м: -0.2 в секунду -> -0.02 за шаг (0.1 с)
        if s["H"] < 30.0:
            reward -= 0.02
            
        # Расстояние до препятствия меньше 15 м: -0.5 в секунду -> -0.05 за шаг
        # Проверим минимальную дальность по лучам или препятствиям
        min_obs_dist = self._get_min_obstacle_dist(s)
        if min_obs_dist < 15.0:
            reward -= 0.05
            
        # Резкость уставок: -0.02 на квадрат изменения уставки
        action_diff = action - self.prev_action
        reward -= 0.02 * np.sum(action_diff ** 2)
        
        # Время: -0.01 в секунду -> -0.001 за шаг
        reward -= 0.001
        
        # Экстренный режим: -0.1 в секунду -> -0.01 за шаг
        if emergency_flag > 0.5:
            reward -= 0.01

        # 3. Условия завершения
        terminated = s["crashed"] or s["goal_dist"] < 30.0
        truncated = self.episode_steps >= self.max_episode_steps
        
        # 4. Обновляем состояние для следующего шага
        self.prev_dist = s["goal_dist"]
        self.prev_motor_on = s["motor_on"]
        self.prev_action = action.copy()
        
        return self._get_obs(s), reward, terminated, truncated, self._get_info(s)

    def _get_min_obstacle_dist(self, s):
        # Упрощенная оценка: берем минимальное значение из 75 лучей * 300.0 (max_range в motoglider.py)
        rays = self.sim.rays()
        return float(np.min(rays)) * 300.0

    def _get_obs(self, s):
        if self.mode == "teacher":
            rays = self.sim.rays()
            wind = self.sim.wind()["world"]
            nav = np.array([
                s["V"], s["alpha"], s["beta"], s["gamma"], s["theta"],
                s["omega"][0], s["omega"][1], s["omega"][2], s["H"], s["battery"]
            ], dtype=np.float32)
            goal_info = np.array([
                s["goal_dist"], 
                math.atan2(s["goal"][0] - s["pos"][0], s["goal"][1] - s["pos"][1]),
                s["goal"][2] - s["H"]
            ], dtype=np.float32)
            # Упрощенный вектор препятствий (ближайшие 5, их скорость)
            obstacles = self.sim.obstacles()
            obs_vec = np.concatenate([rays, wind, nav, goal_info, np.zeros(20, dtype=np.float32)]) # Заглушка для препятствий, можно расширить
            return obs_vec.astype(np.float32)
        else:
            depth = self.sim.depth()
            # Заменяем inf на большое число для стабильности сети
            depth = np.where(np.isinf(depth), 400.0, depth)
            wind = self.sim.wind(noise=0.05)["world"] # Шум 0.05 м/с по ТЗ
            nav = np.array([
                s["V"], s["alpha"], s["beta"], s["gamma"], s["theta"],
                s["omega"][0], s["omega"][1], s["omega"][2], s["H"], s["battery"],
                s["goal_dist"], s["goal"][2] - s["H"], float(s["motor_on"]), float(s["emergency"])
            ], dtype=np.float32)
            return {"depth": depth.astype(np.float32), "vector": nav}

    def _get_info(self, s):
        return {
            "goal_dist": s["goal_dist"],
            "crashed": s["crashed"],
            "reason": s["reason"] if s["crashed"] else "",
            "battery": s["battery"],
            "motor_on": s["motor_on"]
        }

    def render(self):
        if self.render_mode == "human":
            # MuJoCo viewer запускается отдельно, здесь можно просто выводить телеметрию
            pass

    def close(self):
        self.sim.close()