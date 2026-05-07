#!/usr/bin/env python3
import argparse
import signal
import sys
import time
from multiprocessing import Process

import torch


STOP = False


def handle_signal(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep each selected GPU at a small but steady utilization level."
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--target-util",
        type=float,
        default=7.0,
        help="Requested per-GPU average utilization percentage.",
    )
    parser.add_argument(
        "--control-window-ms",
        type=float,
        default=200.0,
        help="Smaller windows make utilization steadier in monitoring tools.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=192,
        help="Working matrix size. Larger values increase pulse strength.",
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def run_worker(device_idx: int, target_util: float, window_ms: float, matrix_size: int, dtype_name: str):
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")
    dtype = torch.float16 if dtype_name == "float16" else torch.float32

    target_util = max(0.0, min(100.0, target_util))
    window_s = max(0.05, window_ms / 1000.0)
    busy_budget = window_s * target_util / 100.0

    size = max(96, matrix_size)
    min_size = 96
    max_size = 768

    # Keep a tiny persistent workload per GPU. This adds only a small amount of memory
    # while avoiding allocator churn and making the utilization signal steadier.
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)

    # Stagger startup so all GPUs do not pulse at the exact same phase.
    time.sleep((device_idx % 8) * 0.017)

    while not STOP:
        loop_start = time.perf_counter()
        busy_start = time.perf_counter()
        iters = 0

        while not STOP and time.perf_counter() - busy_start < busy_budget:
            c = torch.matmul(a, b)
            a = torch.matmul(c, b)
            iters += 2

        torch.cuda.synchronize(device)
        busy_elapsed = time.perf_counter() - busy_start

        # Adapt matrix size so each GPU stays near the requested duty cycle.
        if busy_budget > 0:
            ratio = busy_elapsed / busy_budget
            if ratio < 0.75 and size < max_size:
                size = min(max_size, int(size * 1.12))
                a = torch.randn((size, size), device=device, dtype=dtype)
                b = torch.randn((size, size), device=device, dtype=dtype)
            elif ratio > 1.25 and size > min_size:
                size = max(min_size, int(size / 1.12))
                a = torch.randn((size, size), device=device, dtype=dtype)
                b = torch.randn((size, size), device=device, dtype=dtype)

        elapsed = time.perf_counter() - loop_start
        sleep_for = window_s - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    args = parse_args()

    gpu_ids = [int(part.strip()) for part in args.gpus.split(",") if part.strip()]
    visible = torch.cuda.device_count()
    invalid = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= visible]
    if invalid:
        raise SystemExit(f"Invalid GPU ids {invalid}; visible device count is {visible}")

    workers = []
    for gpu in gpu_ids:
        proc = Process(
            target=run_worker,
            args=(gpu, args.target_util, args.control_window_ms, args.matrix_size, args.dtype),
        )
        proc.start()
        workers.append(proc)

    try:
        while not STOP:
            time.sleep(5)
            for proc in workers:
                if not proc.is_alive():
                    raise RuntimeError(f"Worker process {proc.pid} exited unexpectedly")
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
