# U_Net
CUDA_VISIBLE_DEVICES=0 python pytorch_run.py --model_id 0 --dataseed 1187 --dataset busi

# Att_U_Net
CUDA_VISIBLE_DEVICES=1 python pytorch_run.py --model_id 1 --dataseed 1187 --dataset busi

# U_Net++
CUDA_VISIBLE_DEVICES=2 python pytorch_run.py --model_id 4 --dataseed 1187 --dataset busi


# U_Net
CUDA_VISIBLE_DEVICES=3 python pytorch_run.py --model_id 0 --dataseed 1187 --dataset glas --input_size 512 --batch_size 4 --initial_lr 0.001

# Att_U_Net
CUDA_VISIBLE_DEVICES=4 python pytorch_run.py --model_id 1 --dataseed 1187 --dataset glas --input_size 512 --batch_size 4 --initial_lr 0.001

# U_Net++
CUDA_VISIBLE_DEVICES=5 python pytorch_run.py --model_id 4 --dataseed 1187 --dataset glas --input_size 512 --batch_size 2 --initial_lr 0.001


# U_Net
CUDA_VISIBLE_DEVICES=6 python pytorch_run.py --model_id 0 --dataseed 1187 --dataset chase --input_size 960 --batch_size 2 --initial_lr 0.001

# Att_U_Net
CUDA_VISIBLE_DEVICES=7 python pytorch_run.py --model_id 1 --dataseed 1187 --dataset chase --input_size 960 --batch_size 2 --initial_lr 0.001

# U_Net++
CUDA_VISIBLE_DEVICES=4 python pytorch_run.py --model_id 4 --dataseed 1187 --dataset chase --input_size 960 --batch_size 2 --initial_lr 0.001


# U_Net
CUDA_VISIBLE_DEVICES= python pytorch_run.py --model_id 0 --dataseed 1187 --dataset cvc

# Att_U_Net
CUDA_VISIBLE_DEVICES= python pytorch_run.py --model_id 2 --dataseed 1187 --dataset cvc

# U_Net++
CUDA_VISIBLE_DEVICES= python pytorch_run.py --model_id 4 --dataseed 1187 --dataset cvc


# U_Net
CUDA_VISIBLE_DEVICES=4 python pytorch_run.py --model_id 0 --dataseed 1187 --dataset kvasir

# Att_U_Net
CUDA_VISIBLE_DEVICES= python pytorch_run.py --model_id 2 --dataseed 1187 --dataset kvasir

# U_Net++
CUDA_VISIBLE_DEVICES=7 python pytorch_run.py --model_id 4 --dataseed 1187 --dataset kvasir

# New_UNet
CUDA_VISIBLE_DEVICES=0 python pytorch_run.py --model_id 5 --dataseed 1187 --dataset kvasir
