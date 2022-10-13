#!/bin/bash
# 585k
#for i in `seq 505000 5000 625000`
#CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov5 --past_ref --timing --memory --save_results
#CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov10 --past_ref --timing --memory --save_results
#CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov20 --past_ref --timing --memory --save_results

CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov5 --past_ref --timing --memory --reverse

# Find Best FlowLens Inner
for i in `seq 400000 5000 450000`
do
#  echo $i
  echo gen_${i}.pth
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-I_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//InnerSphere --fov fov5 --past_ref --timing --memory --output_size 336 336 --model_win_size 7 7 --model_output_size 84 84
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-I_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//InnerSphere --fov fov10 --past_ref --timing --memory --output_size 336 336 --model_win_size 7 7 --model_output_size 84 84
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-I_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//InnerSphere --fov fov20 --past_ref --timing --memory --output_size 336 336 --model_win_size 7 7 --model_output_size 84 84
done
