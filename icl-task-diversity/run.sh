#!/usr/bin/env bash
#!/usr/bin/env bash
set -e

git pull

(
  trap 'kill 0' SIGINT

  CUDA_VISIBLE_DEVICES=0 python run.py --config-name=ou_student_fixed_support_weight_sampling task.distrib_param=5.0 &

  sleep 10
  CUDA_VISIBLE_DEVICES=1 python run.py --config-name=ou_student_fixed_support_weight_sampling task.distrib_param=10.0 &

  sleep 20
  CUDA_VISIBLE_DEVICES=3 python run.py --config-name=linreg_student_fixed_support_weight_sampling task.distrib_param=inf,3.0 &

  sleep 30
  CUDA_VISIBLE_DEVICES=2 python run.py --config-name=linreg_student_fixed_support_weight_sampling task.distrib_param=5.0,10.0 &

  wait
)

