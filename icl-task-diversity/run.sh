#!/usr/bin/env bash

set -e

git pull && \
    (trap 'kill 0' SIGINT; \
    (CUDA_VISIBLE_DEVICES=1 python run.py --config-name=plain && python analyze.py) && \
    (sleep 10 && \
    CUDA_VISIBLE_DEVICES=2 python run.py --config-name=plain task.n_tasks=0 && python analyze.py) & wait)
