
pid=3240723
while kill -0 $pid 2>/dev/null; do
    sleep 10
done

CUDA_VISIBLE_DEVICES=2 python run.py --config-name=reweighting_high;
CUDA_VISIBLE_DEVICES=2 python run.py --config-name=reweighting_sampling

