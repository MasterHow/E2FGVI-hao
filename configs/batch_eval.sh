#!/bin/bash
# Find Best FlowLens Inner
#for i in `seq 250000 5000 320000`
#do
##  echo $i
#  echo gen_${i}.pth
#  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-I_FlowLens_early/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//InnerSphere --fov fov5 --past_ref --timing --memory --output_size 336 336 --model_win_size 7 7 --model_output_size 84 84
#  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-I_FlowLens_early/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//InnerSphere --fov fov10 --past_ref --timing --memory --output_size 336 336 --model_win_size 7 7 --model_output_size 84 84
#  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-I_FlowLens_early/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//InnerSphere --fov fov20 --past_ref --timing --memory --output_size 336 336 --model_win_size 7 7 --model_output_size 84 84
#done

#CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov5 --past_ref --timing --memory --reverse --save_results
#CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov10 --past_ref --timing --memory --reverse --save_results
#CUDA_VISIBLE_DEVICES=1 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_large-MFN-T1C1-LastMem-csT-deco-csfv2-Mix-d9/gen_585000.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov20 --past_ref --timing --memory --reverse --save_results

# Find Best FlowLens Outer
#for i in `seq 400000 5000 445000`
#do
##  echo $i
#  echo gen_${i}.pth
#  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_FlowLens_early/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov5 --past_ref --timing --memory
#  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_FlowLens_early/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov10 --past_ref --timing --memory
#  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/lite-MFN_KITTI360EX-O_FlowLens_early/gen_${i}.pth --dataset KITTI360-EX --data_root datasets//KITTI-360EX//OuterPinhole --fov fov20 --past_ref --timing --memory
#done

# Find Best FlowLens Youtube
for i in `seq 370000 2500 377500`
do
#  echo $i
  echo gen_${i}.pth
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --model lite-MFN --ckpt release_model/FlowLens_v2_ubuntu/gen_${i}.pth --dataset davis --data_root datasets/ --timing --memory --same_memory --good_fusion
done
