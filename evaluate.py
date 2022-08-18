# -*- coding: utf-8 -*-
import cv2
import numpy as np
import importlib
import sys
import os
import time
import argparse
import line_profiler
from PIL import Image

import torch
from torch.utils.data import DataLoader

from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model

# global variables
w, h = 432, 240     # default acc. test setting in e2fgvi for davis dataset
# w, h = 864, 480     # davis res 480x854 default speed test setting in e2fgvi for davis dataset
ref_length = 10     # non-local frames的步幅间隔，此处为每10帧取1帧NLF
neighbor_stride = 5     # local frames的窗口大小，加上自身则窗口大小为6
default_fps = 24


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def main_worker(args):
    args.size = (w, h)
    # set up datasets and data loader
    assert (args.dataset == 'davis') or args.dataset == 'youtube-vos', \
        f"{args.dataset} dataset is not supported"
    test_dataset = TestDataset(args)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    if args.ckpt is not None:
        data = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(data)
        print(f'Loading from: {args.ckpt}')
    model.eval()

    total_frame_psnr = []
    total_frame_ssim = []

    output_i3d_activations = []
    real_i3d_activations = []

    print('Start evaluation...')

    if args.timing:
        time_all = 0
        len_all = 0

    # create results directory
    result_path = os.path.join('results', f'{args.model}_{args.dataset}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    eval_summary = open(
        os.path.join(result_path, f"{args.model}_{args.dataset}_metrics.txt"),
        "w")

    i3d_model = init_i3d_model()

    for index, items in enumerate(test_loader):

        frames, masks, video_name, frames_PIL = items

        video_length = frames.size(1)
        frames, masks = frames.to(device), masks.to(device)
        ori_frames = frames_PIL     # 原始帧，可视为真值
        ori_frames = [
            ori_frames[i].squeeze().cpu().numpy() for i in range(video_length)
        ]
        comp_frames = [None] * video_length     # 补全帧

        if args.timing:
            len_all += video_length

        # complete holes by our model
        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                 min(video_length, f + neighbor_stride + 1))
            ]   # neighbor_ids即为Local Frames, 局部帧
            ref_ids = get_ref_index(neighbor_ids, video_length)     # ref_ids即为Non-Local Frames, 非局部帧
            selected_imgs = frames[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_frames = selected_imgs * (1 - selected_masks)

                if args.timing:
                    torch.cuda.synchronize()
                    time_start = time.time()
                pred_img, _ = model(masked_frames, len(neighbor_ids))
                if args.timing:
                    torch.cuda.synchronize()
                    time_end = time.time()
                    time_sum = time_end - time_start
                    time_all += time_sum
                    # print('Run Time: '
                    #       f'{time_sum/len(neighbor_ids)}')

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])

                    if not args.good_fusion:
                        if comp_frames[idx] is None:
                            # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                            comp_frames[idx] = img

                        else:   # default 融合策略：不合理，neighbor_stride倍数的LF的中间帧权重为0.25，应当为0.5
                            # 如果不是第一次补全Local Frame中的某帧，即该帧已补全过，则把此前结果与当前帧结果简单加和平均
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        ########################################################################################
                    else:
                        if comp_frames[idx] is None:
                            # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                            comp_frames[idx] = img

                        elif idx == (neighbor_ids[0] + neighbor_ids[-1])/2:
                            # 如果是中间帧，记录下来
                            medium_frame = img
                        elif (idx != 0) & (idx == neighbor_ids[0]):
                            # 如果是第三次出现，加权平均
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.25 + medium_frame.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.25
                        else:
                            # 如果是不是中间帧，权重为0.5
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        ########################################################################################

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
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}'
        )
        eval_summary.write(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}\n'
        )

        if args.timing:
            print('Average run time: (%f) per frame' % (time_all/len_all))

        # saving images for evaluating warpping errors
        if args.save_results:
            save_frame_path = os.path.join(result_path, video_name[0])
            os.makedirs(save_frame_path, exist_ok=False)

            for i, frame in enumerate(comp_frames):
                cv2.imwrite(
                    os.path.join(save_frame_path,
                                 str(i).zfill(5) + '.png'),
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))

    avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
    avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)

    fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)
    print('Finish evaluation... Average Frame PSNR/SSIM/VFID: '
          f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.write(
        'Finish evaluation... Average Frame PSNR/SSIM/VFID: '
        f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.close()

    if args.timing:
        print('All average forward run time: (%f) per frame' % (time_all / len_all))

    return len(total_frame_psnr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E2FGVI')
    parser.add_argument('--dataset',
                        choices=['davis', 'youtube-vos'],
                        type=str)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model', choices=['e2fgvi', 'e2fgvi_hq', 'e2fgvi_hq-lite', 'lite-MFN'], type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--timing', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--good_fusion', action='store_true', default=False, help='using my fusion strategy')
    args = parser.parse_args()

    if args.profile:
        profile = line_profiler.LineProfiler(main_worker)  # 把函数传递到性能分析器
        profile.enable()  # 开始分析

    # if args.timing:
    #     torch.cuda.synchronize()
    #     time_start = time.time()

    frame_num = main_worker(args)

    # if args.timing:
    #     torch.cuda.synchronize()
    #     time_end = time.time()
    #     time_sum = time_end - time_start
    #     print('Finish evaluation... Average Run Time: '
    #           f'{time_sum/frame_num}')

    if args.profile:
        profile.disable()  # 停止分析
        profile.print_stats(sys.stdout)  # 打印出性能分析结果
