# UNet
# CUDA_VISIBLE_DEVICES=6 python pytorch_run.py --model_id 0 --dataseed 1187 --dataset kvasir

# CUDA_VISIBLE_DEVICES=6 python pytorch_run.py --model_id 0 --dataseed 2981 --dataset kvasir

# CUDA_VISIBLE_DEVICES=6 python pytorch_run.py --model_id 0 --dataseed 6142 --dataset kvasir

# New_UNet
CUDA_VISIBLE_DEVICES=5 python pytorch_run.py --model_id 5 --dataseed 1187 --dataset kvasir

CUDA_VISIBLE_DEVICES=6 python pytorch_run.py --model_id 5 --dataseed 2981 --dataset kvasir

CUDA_VISIBLE_DEVICES=7 python pytorch_run.py --model_id 5 --dataseed 6142 --dataset kvasir
