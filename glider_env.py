# glider_env.py
import gymnasium as gym
import numpy as np
import math
import motoglider as mg

DEG = math.pi / 180.0

class GliderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, mode="student", render_mode=None, randomize=True, difficulty=1):
        super().__init__()
        self.mode = mode
        self.render_mode = render_mode
        self.randomize = randomize
        self.difficulty = difficulty
        
        depth_enabled = (mode == "student")
        self.sim = mg.GliderSim(depth=depth_enabled, depth_size=(64, 40))
        
        # ✅ ПРОСТРАНСТВО ДЕЙСТВИЙ: 6 элементов
        # [крен, скорость, скольжение, throttle_raw, emergency, motor_on]
        self.action_space = gym.spaces.Box(
            low=np.array([-45*DEG, 9.5, -10*DEG, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([45*DEG, 20.0, 10*DEG, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Пространство наблюдений
        if self.mode == "teacher":
            obs_dim = 75 + 3 + 10 + 4 + 16  # 108
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict({
                "depth": gym.spaces.Box(low=0.0, high=400.0, shape=(40, 64), dtype=np.float32),
                "vector": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
            })

        # ✅ ВАЖНО: prev_action теперь 6-мерный!
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.prev_action[1] = 12.0  # Начальная уставка скорости
        self.prev_dist = np.inf
        self.prev_motor_on = False
        self.prev_H = 130.0
        self.episode_steps = 0
        self.max_episode_steps = 1500

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        wind_speed = np.random.uniform(0.0, 7.0)
        wind_dir = np.random.uniform(0.0, 360.0)
        
        if self.difficulty == 1:
            # Очень близкая цель (200-300 м) - агент ТОЧНО долетит в планировании
            dist = np.random.uniform(200.0, 300.0)
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            start = (0.0, 0.0, 130.0)
            goal = (dist * np.cos(angle), dist * np.sin(angle), 110.0)
            heading = 90.0
            s = self.sim.reset(start=start, goal=goal, heading_deg=heading, wind=(0.0, 0.0))
        elif self.difficulty == 2:
            # Средняя дистанция (400-700 м)
            dist = np.random.uniform(400.0, 700.0)
            angle = np.random.uniform(-np.pi/3, np.pi/3)
            start = (0.0, 0.0, 130.0)
            goal = (dist * np.cos(angle), dist * np.sin(angle), np.random.uniform(90.0, 130.0))
            heading = 90.0
            wind_speed = np.random.uniform(0.0, 3.0)
            wind_dir = np.random.uniform(0.0, 360.0)
            s = self.sim.reset(start=start, goal=goal, heading_deg=heading, wind=(wind_speed, wind_dir))
        elif self.difficulty == 3:
            # Дальняя дистанция (800-1500 м)
            dist = np.random.uniform(800.0, 1500.0)
            angle = np.random.uniform(-np.pi/2, np.pi/2)
            start = (0.0, 0.0, 130.0)
            goal = (dist * np.cos(angle), dist * np.sin(angle), np.random.uniform(80.0, 150.0))
            heading = 90.0
            wind_speed = np.random.uniform(0.0, 5.0)
            wind_dir = np.random.uniform(0.0, 360.0)
            s = self.sim.reset(start=start, goal=goal, heading_deg=heading, wind=(wind_speed, wind_dir))
        else:
            # Полная рандомизация
            if self.randomize:
                start, goal, heading = self.sim.randomize_scene(rng=np.random.default_rng(seed))
                s = self.sim.reset(start=start, goal=goal, heading_deg=heading, wind=(wind_speed, wind_dir))
            else:
                s = self.sim.reset(wind=(wind_speed, wind_dir))

        # ✅ Инициализация 6-мерного prev_action
        self.prev_action = np.array([0.0, 12.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_dist = s["goal_dist"]
        self.prev_motor_on = s["motor_on"]
        self.prev_H = s["H"]
        self.episode_steps = 0
        
        return self._get_obs(s), self._get_info(s)

    def step(self, action):
        self.episode_steps += 1
        
        # ✅ Разбираем 6-мерное действие
        gamma, v_ref, beta, throttle_raw, emergency_flag, motor_on_flag = action
        
        # ✅ ЛОГИКА БИНАРНОГО УПРАВЛЕНИЯ МОТОРОМ:
        # Если motor_on_flag < 0.5 - мотор выключен (throttle = 0)
        # Если motor_on_flag >= 0.5 - мотор включён, throttle >= 0.3 (избегаем "мёртвой зоны")
        if motor_on_flag < 0.5:
            throttle = 0.0
        else:
            throttle = max(0.3, throttle_raw)
        
        elevator = None
        if emergency_flag > 0.5:
            elevator = ((v_ref - 9.5) / (20.0 - 9.5)) * 30.0 * DEG - 15.0 * DEG
            v_ref = 12.0

        s = self.sim.step(gamma, v_ref, beta, throttle, elevator=elevator)

        # Вычисляем награду
        reward = 0.0
        
        # Приближение к цели: +0.1 за метр
        dist_change = self.prev_dist - s["goal_dist"]
        reward += 0.1 * dist_change
        
        # Достижение цели: +200
        if s["goal_dist"] < 30.0:
            reward += 200.0
            
        # Столкновение/разрушение/сваливание: -10
        if s["crashed"]:
            reward -= 10.0
        
        # ✅ ШТРАФ ЗА НЕЭФФЕКТИВНУЮ РАБОТУ МОТОРА
        # Если motor_on_flag > 0.5 (хотели включить), но тяга = 0 - это плохо
        if motor_on_flag > 0.5 and s["thrust"] < 0.1:
            reward -= 0.5
        
        # ✅ ШТРАФ ЗА ВКЛЮЧЕНИЕ МОТОРА КОГДА НЕ НУЖНО
        # Если мотор включён, но мы не приближаемся к цели - штраф
        if motor_on_flag > 0.5 and dist_change < 0:
            reward -= 0.1
        
        # Энергия мотора: -0.01 за кДж
        energy_kj = s["power"] * 0.1 / 1000.0
        reward -= 0.01 * energy_kj
        
        # Включение мотора: -0.05
        if s["motor_on"] and not self.prev_motor_on:
            reward -= 0.05
            
        # Высота меньше 30 м: -0.02 за шаг
        if s["H"] < 30.0:
            reward -= 0.02
            
        # Расстояние до препятствия меньше 15 м: -0.05 за шаг
        min_obs_dist = self._get_min_obstacle_dist(s)
        if min_obs_dist < 15.0:
            reward -= 0.05
        
        # Бонус за приближение к цели (когда близко)
        if s["goal_dist"] < 100.0:
            reward += 0.5
        if s["goal_dist"] < 50.0:
            reward += 1.0
        
        # Штраф за трату высоты без приближения к цели
        if dist_change < 0 and s["H"] < self.prev_H - 2.0:
            reward -= 0.1
        
        # Резкость уставок (только первые 5 элементов, не считая motor_on)
        action_diff = action[:5] - self.prev_action[:5]
        reward -= 0.02 * np.sum(action_diff ** 2)
        
        # Время: -0.001 за шаг
        reward -= 0.001
        
        # Экстренный режим: -0.01 за шаг
        if emergency_flag > 0.5:
            reward -= 0.01

        # Условия завершения
        terminated = s["crashed"] or s["goal_dist"] < 30.0
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Обновляем состояние
        self.prev_dist = s["goal_dist"]
        self.prev_motor_on = s["motor_on"]
        self.prev_action = action.copy()
        self.prev_H = s["H"]
        
        return self._get_obs(s), reward, terminated, truncated, self._get_info(s)

    def _get_min_obstacle_dist(self, s):
        rays = self.sim.rays()
        return float(np.min(rays)) * 300.0

    def _get_obs(self, s):
        if self.mode == "teacher":
            rays = self.sim.rays()
            wind = self.sim.wind()["world"]
            
            nav = np.array([
                s["V"] / 20.0,
                s["alpha"] / (15 * DEG),
                s["beta"] / (10 * DEG),
                s["gamma"] / (45 * DEG),
                s["theta"] / (25 * DEG),
                s["omega"][0] / 2.0,
                s["omega"][1] / 2.0,
                s["omega"][2] / 2.0,
                s["H"] / 200.0,
                s["battery"]
            ], dtype=np.float32)
            
            goal_vec = s["goal"][:2] - s["pos"][:2]
            goal_bearing = math.atan2(goal_vec[0], goal_vec[1])
            rel_bearing = (goal_bearing - s["heading"] + math.pi) % (2 * math.pi) - math.pi
            
            goal_info = np.array([
                s["goal_dist"] / 2000.0,
                math.sin(rel_bearing),
                math.cos(rel_bearing),
                (s["goal"][2] - s["H"]) / 100.0
            ], dtype=np.float32)
            
            wind_body = np.array([
                s["headwind"] / 10.0,
                s["crosswind"] / 10.0,
                0.0
            ], dtype=np.float32)
            
            obs_vec = np.concatenate([rays, wind_body, nav, goal_info, np.zeros(16, dtype=np.float32)])
            return obs_vec.astype(np.float32)
        else:
            depth = self.sim.depth()
            depth = np.where(np.isinf(depth), 400.0, depth)
            wind = self.sim.wind(noise=0.05)
            wind_body = np.array([
                wind["headwind"] / 10.0,
                wind["crosswind"] / 10.0,
                0.0
            ], dtype=np.float32)
            
            goal_vec = s["goal"][:2] - s["pos"][:2]
            goal_bearing = math.atan2(goal_vec[0], goal_vec[1])
            rel_bearing = (goal_bearing - s["heading"] + math.pi) % (2 * math.pi) - math.pi
            
            nav = np.array([
                s["V"] / 20.0,
                s["alpha"] / (15 * DEG),
                s["beta"] / (10 * DEG),
                s["gamma"] / (45 * DEG),
                s["theta"] / (25 * DEG),
                s["omega"][0] / 2.0,
                s["omega"][1] / 2.0,
                s["omega"][2] / 2.0,
                s["H"] / 200.0,
                s["battery"],
                s["goal_dist"] / 2000.0,
                math.sin(rel_bearing),
                math.cos(rel_bearing),
                (s["goal"][2] - s["H"]) / 100.0
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
        pass

    def close(self):
        self.sim.close()