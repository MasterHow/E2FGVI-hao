#!/bin/bash
# Find Best FlowLens Inner
#for i in `seq 400000 5000 495000`
#do
##  echo $i
#  echo gen_${i}.pth
#  CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov5 --past_ref --timing --memory
#  CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov10 --past_ref --timing --memory
#  CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov20 --past_ref --timing --memory
#done

CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov5 --past_ref --timing --memory --reverse --save_results
CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov10 --past_ref --timing --memory --reverse --save_results
CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov20 --past_ref --timing --memory --reverse --save_results

