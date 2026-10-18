# watch_teacher_visual.py
import argparse
import math
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import mujoco
import motoglider as mg
from stable_baselines3 import PPO

D = math.pi / 180.0

class NeuralPolicy:
    """Обученная нейросеть с 6-мерным action space."""
    def __init__(self, model_path, mode="teacher"):
        self.model = PPO.load(model_path)
        self.mode = mode
        self.last_action = None
        
    def __call__(self, sim, s):
        # Формируем наблюдение точно так же, как в glider_env.py
        rays = sim.rays()
        
        nav = np.array([
            s["V"] / 20.0,
            s["alpha"] / (15 * D),
            s["beta"] / (10 * D),
            s["gamma"] / (45 * D),
            s["theta"] / (25 * D),
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
        
        # ✅ Объединяем наблюдения (размер 108)
        obs = np.concatenate([rays, wind_body, nav, goal_info, np.zeros(16, dtype=np.float32)])
        
        # Получаем действие от модели
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        self.last_action = action.copy()
        
        # ✅ РАЗБОР 6-МЕРНОГО ДЕЙСТВИЯ
        gamma, v_ref, beta, throttle_raw, emergency_flag, motor_on_flag = action
        
        # ✅ ЛОГИКА БИНАРНОГО УПРАВЛЕНИЯ МОТОРОМ
        if motor_on_flag < 0.5:
            throttle = 0.0
        else:
            throttle = max(0.3, throttle_raw)
        
        # Если экстренный режим, преобразуем скорость в угол РВ
        if emergency_flag > 0.5:
            elevator = ((v_ref - 9.5) / (20.0 - 9.5)) * 30.0 * D - 15.0 * D
            return gamma, 12.0, beta, throttle, elevator, motor_on_flag
        
        return gamma, v_ref, beta, throttle, motor_on_flag

def choose_setpoints(policy, sim, s):
    result = policy(sim, s)
    if len(result) == 6:
        gamma, v_ref, beta, throttle, elevator, motor_on = result
        return sim.step(gamma, v_ref, beta, throttle, elevator=elevator)
    else:
        gamma, v_ref, beta, throttle, motor_on = result
        return sim.step(gamma, v_ref, beta, throttle)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./logs/teacher_curriculum_v6/best_model/best_model",
                    help="Путь к обученной модели")
    ap.add_argument("--wind", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=1, help="Количество эпизодов")
    ap.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости")
    ARGS = ap.parse_args()
    
    # Загружаем модель
    if not os.path.exists(ARGS.model + ".zip"):
        print(f"Модель не найдена: {ARGS.model}.zip")
        print("Запустите обучение или укажите правильный путь.")
        return 1
    
    print(f"📂 Загрузка модели: {ARGS.model}")
    policy = NeuralPolicy(ARGS.model)
    
    for ep in range(ARGS.episodes):
        print(f"\n{'='*70}")
        print(f"🚀 ЭПИЗОД {ep + 1}/{ARGS.episodes}")
        print(f"{'='*70}")
        
        # Инициализация симулятора
        sim = mg.GliderSim()
        
        # Рандомизация сцены
        if ARGS.seed is not None:
            rng = np.random.default_rng(ARGS.seed + ep)
        else:
            rng = np.random.default_rng()
        
        start, goal, heading = sim.randomize_scene(rng=rng)
        wind_speed = rng.uniform(0.0, 5.0)
        wind_dir = rng.uniform(0.0, 360.0)
        
        s = sim.reset(start=start, goal=goal, heading_deg=heading, wind=(wind_speed, wind_dir))
        
        print(f"📍 Старт: ({start[0]:.0f}, {start[1]:.0f}, {start[2]:.0f}) м")
        print(f"🎯 Цель: ({goal[0]:.0f}, {goal[1]:.0f}, {goal[2]:.0f}) м")
        print(f"🧭 Курс: {heading:.0f}°")
        print(f"💨 Ветер: {wind_speed:.1f} м/с, направление {wind_dir:.0f}°")
        print(f"📏 Дистанция: {s['goal_dist']:.0f} м")
        print("Управление: q - выход")
        
        from mujoco import viewer as mj_viewer
        
        with mj_viewer.launch_passive(sim.model, sim.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = sim.bid
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 14.0, -15.0, 180.0
            
            while v.is_running():
                t0 = time.time()
                with v.lock():
                    s = choose_setpoints(policy, sim, s)
                
                # Подробная телеметрия каждые 2 секунды
                if int(sim.t * 10) % 20 == 0:
                    action = policy.last_action
                    if action is not None:
                        gamma, v_ref, beta, throttle_raw, emergency, motor_on = action
                        motor_on_str = "ВКЛ" if motor_on > 0.5 else "выкл"
                        throttle = 0.0 if motor_on < 0.5 else max(0.3, throttle_raw)
                        print(f"t={sim.t:6.1f}с  dist={s['goal_dist']:5.0f}м  H={s['H']:5.0f}м  "
                              f"V={s['V']:4.1f}м/с  "
                              f"мотор={motor_on_str}  throttle={throttle:.2f}  "
                              f"батарея={s['battery']*100:3.0f}%  "
                              f"мощность={s['power']/1000:.2f}кВт  "
                              f"тяга={s['thrust']:.1f}Н", flush=True)
                    else:
                        print(f"t={sim.t:6.1f}с  dist={s['goal_dist']:5.0f}м  H={s['H']:5.0f}м  "
                              f"V={s['V']:4.1f}м/с  батарея={s['battery']*100:3.0f}%", flush=True)
                
                if s["crashed"]:
                    print(f"\n💥 [АВАРИЯ] {s['reason']} на t={sim.t:.1f}с")
                    print(f"   Финальная дистанция: {s['goal_dist']:.0f} м")
                    print(f"   Остаток батареи: {s['battery']*100:.1f}%")
                    print(f"   Расход энергии мотора: {s['E_motor']/1000:.2f} кДж")
                    break
                elif s["goal_dist"] < 30:
                    print(f"\n✅ [ЦЕЛЬ ДОСТИГНУТА] на t={sim.t:.1f}с")
                    print(f"   Остаток батареи: {s['battery']*100:.1f}%")
                    print(f"   Расход энергии мотора: {s['E_motor']/1000:.2f} кДж")
                    break
                
                v.sync()
                time.sleep(max(0.0, 0.1 / max(ARGS.speed, 0.05) - (time.time() - t0)))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())