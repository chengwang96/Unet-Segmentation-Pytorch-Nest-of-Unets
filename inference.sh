# UNet
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 0 --dataseed 1187 --dataset kvasir --log_dir ./output/U_Net_cvc_1187
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 0 --dataseed 2981 --dataset kvasir --log_dir ./output/U_Net_cvc_2981
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 0 --dataseed 6142 --dataset kvasir --log_dir ./output/U_Net_cvc_6142

# New_UNet
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 2981 --dataset busi --log_dir ./output/New_UNet_busi_2981_06012352
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 6142 --dataset busi --log_dir ./output/New_UNet_busi_6142_06012352

CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 1187 --dataset glas --log_dir ./output/New_UNet_glas_1187_06012358 --input_size 512
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 2981 --dataset glas --log_dir ./output/New_UNet_glas_2981_06012358
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 6142 --dataset glas --log_dir ./output/New_UNet_glas_6142_06012358 --input_size 512

CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 2981 --dataset cvc --log_dir ./output/New_UNet_cvc_2981_06020232

CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 5 --dataseed 2981 --dataset kvasir --log_dir ./output/New_UNet_kvasir_2981

# AttU
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 2 --dataseed 6142 --dataset busi --log_dir ./output/AttU_Net_busi_6142
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 2 --dataseed 6142 --dataset glas --log_dir ./output/AttU_Net_glas_2981_06020232  --input_size 512
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 2 --dataseed 2981 --dataset cvc --log_dir ./output/AttU_Net_cvc_2981
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 2 --dataseed 2981 --dataset kvasir --log_dir ./output/AttU_Net_kvasir_2981

# UNet++
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 4 --dataseed 6142 --dataset busi --log_dir ./output/U_Net++_busi_6142
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 4 --dataseed 6142 --dataset glas --log_dir ./output/U_Net++_glas_2981_06020232  --input_size 512
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 4 --dataseed 2981 --dataset cvc --log_dir ./output/U_Net++_cvc_2981
CUDA_VISIBLE_DEVICES=2 python pytorch_inference.py --model_id 4 --dataseed 2981 --dataset kvasir --log_dir ./output/U_Net++_kvasir_2981