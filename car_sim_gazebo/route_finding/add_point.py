#!/usr/bin/env python3
"""
Simple CSV path resampler.

입력 CSV 형식: time_sec,x,y,z,yaw_rad
간격(m)을 지정해 균일한 포인트로 보간한 뒤 동일한 헤더로 저장합니다.

사용 예:
  python3 add_point.py --input /home/yoo/Final_P/car_sim_path.csv \
                       --output /home/yoo/Final_P/car_sim_path_resampled.csv \
                       --spacing 0.02
"""
import argparse
import csv
import math
from typing import List, Tuple


def unwrap_yaw(yaws: List[float]) -> List[float]:
    """Make yaw sequence monotonic (no 2*pi jumps)."""
    if not yaws:
        return yaws
    unwrapped = [yaws[0]]
    for y in yaws[1:]:
        dy = y - unwrapped[-1]
        while dy > math.pi:
            dy -= 2 * math.pi
        while dy < -math.pi:
            dy += 2 * math.pi
        unwrapped.append(unwrapped[-1] + dy)
    return unwrapped


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def load_path(path: str) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    yaws: List[float] = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 5:
                continue
            t, x, y, z, yaw = map(float, row[:5])
            times.append(t)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            yaws.append(yaw)
    return times, xs, ys, zs, yaws


def compute_cumulative_distance(xs: List[float], ys: List[float]) -> List[float]:
    s = [0.0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        ds = math.hypot(x1 - x0, y1 - y0)
        s.append(s[-1] + ds)
    return s


def resample(
    times: List[float],
    xs: List[float],
    ys: List[float],
    zs: List[float],
    yaws: List[float],
    spacing: float,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    if len(xs) < 2:
        return times, xs, ys, zs, yaws

    yaws_unwrapped = unwrap_yaw(yaws)
    s = compute_cumulative_distance(xs, ys)
    total_length = s[-1]
    if total_length <= spacing:
        return times, xs, ys, zs, yaws_unwrapped

    target_s = [i * spacing for i in range(int(total_length / spacing) + 1)]
    if target_s[-1] < total_length:
        target_s.append(total_length)

    out_t: List[float] = []
    out_x: List[float] = []
    out_y: List[float] = []
    out_z: List[float] = []
    out_yaw: List[float] = []

    idx = 0
    for ts in target_s:
        while idx + 1 < len(s) and s[idx + 1] < ts:
            idx += 1
        if idx + 1 >= len(s):
            idx = len(s) - 2
        s0, s1 = s[idx], s[idx + 1]
        ratio = 0.0 if s1 - s0 < 1e-6 else (ts - s0) / (s1 - s0)

        out_t.append(lerp(times[idx], times[idx + 1], ratio))
        out_x.append(lerp(xs[idx], xs[idx + 1], ratio))
        out_y.append(lerp(ys[idx], ys[idx + 1], ratio))
        out_z.append(lerp(zs[idx], zs[idx + 1], ratio))
        out_yaw.append(lerp(yaws_unwrapped[idx], yaws_unwrapped[idx + 1], ratio))

    return out_t, out_x, out_y, out_z, out_yaw


def save_path(path: str, t, x, y, z, yaw):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "x", "y", "z", "yaw_rad"])
        for row in zip(t, x, y, z, yaw):
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Resample path CSV at fixed spacing (meters).")
    parser.add_argument("--input", "-i", required=True, help="Input CSV path (time_sec,x,y,z,yaw_rad)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument(
        "--spacing", "-s", type=float, default=0.01, help="Resample spacing in meters (default: 0.01)"
    )
    args = parser.parse_args()

    times, xs, ys, zs, yaws = load_path(args.input)
    if len(xs) < 2:
        print("Input path has fewer than 2 points; nothing to resample.")
        save_path(args.output, times, xs, ys, zs, yaws)
        return

    t_r, x_r, y_r, z_r, yaw_r = resample(times, xs, ys, zs, yaws, args.spacing)
    save_path(args.output, t_r, x_r, y_r, z_r, yaw_r)
    print(f"Resampled {len(xs)} -> {len(x_r)} points (spacing={args.spacing} m)")


if __name__ == "__main__":
    main()
