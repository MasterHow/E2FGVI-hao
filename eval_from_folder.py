# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import argparse
from PIL import Image
import glob

import torch
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model

# global variables
# w h can be changed by args.output_size
w, h = 432, 240     # default acc. test setting in e2fgvi for davis dataset and KITTI-EXO
# w, h = 336, 336     # default acc. test setting for KITTI-EXI


def main_worker(args):
    w = args.output_size[0]
    h = args.output_size[1]
    args.size = (w, h)

    # 读取视频list
    video_list = []
    for root, dirs, files in os.walk(args.complete_root, topdown=False):
        # 获取文件夹名字
        video_list = dirs

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_frame_psnr = []
    total_frame_ssim = []

    output_i3d_activations = []
    real_i3d_activations = []

    print('Start evaluation...')

    # create results directory
    if args.fov is not None:
        result_path = os.path.join('results', f'{args.model}_{args.fov}_{args.dataset}')
    else:
        result_path = os.path.join('results', f'{args.model}_{args.dataset}')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    eval_summary = open(
        os.path.join(result_path, f"{args.model}_{args.dataset}_metrics.txt"),
        "w")

    i3d_model = init_i3d_model()

    # 评估循环
    for v in range(len(video_list)):

        # 取出一个video地址
        video = video_list[v]

        # frame_dir直接用根目录+视频名字就可以了
        frame_dir = os.path.join(args.complete_root, video)

        # 真值的地址
        gt_frame_dir = os.path.join(args.gt_root, video)

        # 我们的图像是png格式
        frame_list = glob.glob(os.path.join(frame_dir, "*.png"))
        # gt_frame_list = glob.glob(os.path.join(gt_frame_dir, "*.jpg"))

        video_length = len(frame_list)
        comp_frames = []    # 补全帧
        ori_frames = []     # 原始帧
        # 当前视频的工作循环
        for t in range(0, len(frame_list)):
            video_name = video.split('//')[-1]

            ### load input images
            # 我们是png格式的哦
            # filename = os.path.join(frame_dir, "%05d.png" % t)    # misf
            filename = os.path.join(frame_dir, "%08d.png" % t)     # lama, put
            comp_frame = cv2.imread(filename)
            comp_frame = cv2.resize(comp_frame, (w, h))
            comp_frames.append(comp_frame)
            filename = os.path.join(gt_frame_dir, "%06d.jpg" % t)     # misf
            # filename = os.path.join(gt_frame_dir, "%08d.jpg" % t)       # lama
            gt_frame = cv2.imread(filename)
            # resize gt
            gt_frame = cv2.resize(gt_frame, (w, h))
            ori_frames.append(gt_frame)


        # frames = frames.to(device)

        # calculate metrics
        cur_video_psnr = []
        cur_video_ssim = []
        comp_PIL = []  # to calculate VFID
        frames_PIL = []
        for ori, comp in zip(ori_frames, comp_frames):
            psnr, ssim = calc_psnr_and_ssim(ori, comp)

            cur_video_psnr.append(psnr)
            cur_video_ssim.append(ssim)

            total_frame_psnr.append(psnr)
            total_frame_ssim.append(ssim)

            frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
            comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))
        cur_psnr = sum(cur_video_psnr) / len(cur_video_psnr)
        cur_ssim = sum(cur_video_ssim) / len(cur_video_ssim)

        # saving i3d activations
        frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL,
                                                         comp_PIL,
                                                         i3d_model,
                                                         device=device)
        real_i3d_activations.append(frames_i3d)
        output_i3d_activations.append(comp_i3d)

        print(
            f'[{v+1:3}/{len(video_list)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}'
        )
        eval_summary.write(
            f'[{v+1:3}/{len(video_list)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}\n'
        )

    avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
    avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)

    fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)
    print('Finish evaluation... Average Frame PSNR/SSIM/VFID: '
          f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.write(
        'Finish evaluation... Average Frame PSNR/SSIM/VFID: '
        f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.close()

    return len(total_frame_psnr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FlowLens')
    parser.add_argument('--dataset', type=str)       # 相当于train的‘name’
    parser.add_argument('--gt_root', type=str, required=True)
    parser.add_argument('--complete_root', type=str, required=True)

    parser.add_argument('--output_size', type=int, nargs='+', default=[432, 240])
    parser.add_argument('--fov',
                        type=str)  # 对于KITTI360-EX, 测试需要输入fov
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    frame_num = main_worker(args)
