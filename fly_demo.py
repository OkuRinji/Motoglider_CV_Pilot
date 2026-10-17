# -*- coding: utf-8 -*-
"""
fly_demo.py

Полет мотопланера в окне MuJoCo. Управляет простой базовый навигатор без нейросети:
курс на цель, отворот от препятствий по лучам, мотор при нехватке высоты. Он нужен
только для показа модели и как точка сравнения для обученной сети. Место, куда
подставляется сеть, это функция choose_setpoints.

Цель: три числа X Y Z в консоли и Enter (метры; x на восток, y на север, z вверх)
или в окне двойной щелчок по красной сфере и перетаскивание правой кнопкой с Ctrl.

  python fly_demo.py
  python fly_demo.py --wind 5 270
  python fly_demo.py --headless --duration 60
"""

import argparse
import math
import os
import sys
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(description="Полет мотопланера к цели в MuJoCo")
ap.add_argument("--wind", nargs=2, type=float, default=[0.0, 0.0], metavar=("СКОРОСТЬ", "ОТКУДА"))
ap.add_argument("--random-obstacles", action="store_true", help="случайные фазы и скорости движущихся препятствий")
ap.add_argument("--speed", type=float, default=1.0, help="ускорение времени")
ap.add_argument("--headless", action="store_true", help="без окна, только телеметрия")
ap.add_argument("--duration", type=float, default=120.0, help="длительность без окна, с")
ARGS = ap.parse_args()
if ARGS.headless and sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import motoglider as mg  # noqa: E402

D = math.pi / 180.0


class Baseline:
    """Простой навигатор: не решение задачи, а точка сравнения."""

    def __init__(self):
        self.motor = False
        self.avoid = 0.0

    def __call__(self, sim, s):
        goal, pos = s["goal"], s["pos"]
        dv = goal[:2] - pos[:2]
        dist = float(np.linalg.norm(dv))
        err = (math.atan2(dv[0], dv[1]) - s["heading"] + math.pi) % (2 * math.pi) - math.pi
        gamma = max(-35 * D, min(35 * D, err))
        rays = sim.rays().reshape(5, 15) * 300.0
        band = rays[2:4]
        az = np.linspace(-45, 45, 15)
        if band[:, np.abs(az) <= 30].min() < 170 and dist > 60:
            if self.avoid == 0.0:
                self.avoid = -1.0 if band[:, az < 0].mean() > band[:, az > 0].mean() else 1.0
            gamma = self.avoid * 40 * D
        else:
            self.avoid = 0.0
        Kz = 15.0 * max(0.3, 1 - max(s["headwind"], 0) / 12)
        need = goal[2] + dist / Kz + 15
        if pos[2] < need or pos[2] < 40:
            self.motor = True
        if self.motor and pos[2] > need + 25 and pos[2] > 70:
            self.motor = False
        if dist < 80:
            self.motor = pos[2] < goal[2] - 5
        V_ref = 13.0 if self.motor else (11.5 if dist < 80 else 12.0 + 0.5 * max(s["headwind"], 0))
        return gamma, V_ref, 0.0, 1.0 if self.motor else 0.0


def choose_setpoints(policy, sim, s):
    """Сюда подставляется обученная сеть: уставки крена, скорости, скольжения и газа."""
    return policy(sim, s)


def console(sim, lock, stop):
    for line in sys.stdin:
        p = line.replace(",", " ").split()
        if not p:
            continue
        if p[0].lower() in ("q", "quit", "exit"):
            stop.set()
            break
        if p[0].lower() == "s":
            with lock:
                sim.spawn_obstacle(80.0)
            print("[консоль] внезапное препятствие в 80 м по курсу", flush=True)
            continue
        try:
            xyz = [float(v) for v in p[:3]]
            with lock:
                sim.set_goal(xyz)
            print(f"[консоль] цель -> ({xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f})", flush=True)
        except (ValueError, IndexError):
            print("[консоль] нужно три числа: X Y Z", flush=True)


def trail_geoms(scn, pts):
    scn.ngeom = 0
    col = np.array([1.0, 0.65, 0.1, 1.0], dtype=np.float32)
    for i in range(len(pts) - 1):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), col)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, np.asarray(pts[i], float), np.asarray(pts[i + 1], float))
        scn.ngeom += 1


def main():
    sim = mg.GliderSim()
    opts = dict(wind=tuple(ARGS.wind), randomize_obstacles=ARGS.random_obstacles)
    s = sim.reset(**opts)
    pol = Baseline()
    lock, stop = threading.Lock(), threading.Event()
    threading.Thread(target=console, args=(sim, lock, stop), daemon=True).start()
    print("Цель: три числа X Y Z и Enter (x на восток, y на север, z вверх); s: внезапное препятствие; q: выход. Цель:", s["goal"].round(1))
    trail = deque(maxlen=900)
    nxt = dict(p=0.0, tr=0.0)

    def advance():
        nonlocal s
        with lock:
            s = sim.step(*choose_setpoints(pol, sim, s))
        if s["t"] >= nxt["tr"]:
            trail.append(s["pos"].copy())
            nxt["tr"] = s["t"] + 0.5
        if s["t"] >= nxt["p"]:
            print(f"t={s['t']:6.1f} с  x={s['pos'][0]:7.0f} y={s['pos'][1]:6.0f} H={s['H']:5.0f} м  V={s['V']:4.1f} м/с  "
                  f"Vy={s['vel'][2]:+5.2f}  крен {s['gamma'] / D:+5.1f}  мотор {'ВКЛ ' if s['motor_on'] else 'выкл'}  "
                  f"заряд {s['battery'] * 100:3.0f} %  до цели {s['goal_dist']:5.0f} м", flush=True)
            nxt["p"] = s["t"] + 1.0
        if s["crashed"] or s["goal_dist"] < 30:
            print(f"[полет] {'авария: ' + s['reason'] if s['crashed'] else 'цель достигнута'} на {s['t']:.0f} с", flush=True)
            if s["crashed"]:
                goal = s["goal"]
                with lock:
                    s = sim.reset(**opts)
                    sim.set_goal(goal)
                trail.clear()
                nxt.update(p=0.0, tr=0.0)

    if ARGS.headless:
        scn = mujoco.MjvScene(sim.model, maxgeom=2000)
        while sim.t < ARGS.duration and not stop.is_set():
            advance()
            trail_geoms(scn, list(trail))
        print(f"[проверка] шагов {int(round(sim.t / 0.1))}, линий следа {scn.ngeom}")
        return 0
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.bid
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 14.0, -15.0, 180.0
        while v.is_running() and not stop.is_set():
            t0 = time.time()
            with v.lock():
                advance()
                trail_geoms(v.user_scn, list(trail))
            v.sync()
            time.sleep(max(0.0, 0.1 / max(ARGS.speed, 0.05) - (time.time() - t0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
