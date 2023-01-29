# -*- coding: utf-8 -*-
import cv2
import numpy as np
import importlib
import sys
import os
import time
import json
import random
import argparse
# import line_profiler
from PIL import Image

import torch
from torch.utils.data import DataLoader
from mmcv.cnn import get_model_complexity_info

from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model, get_flops

# global variables
# w h can be changed by args.output_size
w, h = 432, 240     # default acc. test setting in e2fgvi for davis dataset and KITTI-EXO
# w, h = 336, 336     # default acc. test setting for KITTI-EXI
# w, h = 864, 480     # davis res 480x854
# w, h = 320, 240     # pal test
ref_length = 10     # non-local frames的步幅间隔，此处为每10帧取1帧NLF
neighbor_stride = 5     # 窗口的stride, 5 for default, 1 for recurrent mode


def read_cfg(args):
    """read lite-MFN cfg from config file"""
    # loading configs
    config = json.load(open(args.cfg_path))

    # # # # pass config to args # # # #
    # REPLACE OLD ARGS
    args.dataset = config['train_data_loader']['name']
    args.data_root = config['train_data_loader']['data_root']
    args.output_size = [432, 240]
    args.output_size[0], args.output_size[1] = (config['train_data_loader']['w'], config['train_data_loader']['h'])
    args.model_win_size = config['model'].get('window_size', None)
    args.model_output_size = config['model'].get('output_size', None)

    args.max_mem_len = config['model'].get('max_mem_len', 1)
    args.compression_factor = config['model'].get('compression_factor', 1)

    # 是否使用spynet作为光流补全网络，目前仅用于消融实验
    config['model']['spy_net'] = config['model'].get('spy_net', 0)
    if config['model']['spy_net'] != 0:
        # default for FlowLens-S
        args.spy_net = True
    else:
        # default for FlowLens
        args.spy_net = False

    args.flow_res = config['model'].get('flow_res', 0.25)  # 默认在1/4分辨率计算光流

    if config['model']['memory'] != 0:
        args.memory = True
    else:
        args.memory = False

    if config['model']['net'] == 'lite-MFN' or config['model']['net'] == 'large-MFN':

        config['model']['fusion_recurrent'] = config['model'].get('fusion_recurrent', 0)  # 默认不循环融合特征
        if config['model']['fusion_recurrent'] != 0:
            args.fusion_recurrent = True
        else:
            # default manner
            args.fusion_recurrent = False

        # 加载E2FGVI-HQ的预训练权重
        config['model']['load_e2fgvi'] = config['model'].get('load_e2fgvi', 0)
        if config['model']['load_e2fgvi'] != 0:
            args.load_e2fgvi = True
        else:
            # default
            args.load_e2fgvi = False

        if config['model']['skip_dcn'] != 0:
            args.skip_dcn = True
        else:
            args.skip_dcn = False

        if config['model']['flow_guide'] != 0:
            args.flow_guide = True
        else:
            args.flow_guide = False

        if config['model']['token_fusion'] != 0:
            args.token_fusion = True
        else:
            args.token_fusion = False

        if config['model']['token_fusion_simple'] != 0:
            args.token_fusion_simple = True
        else:
            args.token_fusion_simple = False

        if config['model']['fusion_skip_connect'] != 0:
            args.fusion_skip_connect = True
        else:
            args.fusion_skip_connect = False

        if args.memory:
            # 额外输入记忆力需要的参数
            # 是否使用空间池化压缩记忆缓存
            if config['model']['mem_pool'] != 0:
                args.mem_pool = True
            else:
                args.mem_pool = False

            # 是否仅存储局部帧的记忆kv
            if config['model']['store_lf'] != 0:
                args.store_lf = True
            else:
                args.store_lf = False

            # 是否在增强前对齐缓存和当前帧的kv
            if config['model']['align_cache'] != 0:
                args.align_cache = True
            else:
                args.align_cache = False

            # 是否在对齐时对token通道分组进行，来实现sub-token的对齐
            if config['model']['sub_token_align'] != 0:
                args.sub_token_align = True
                args.sub_factor = config['model']['sub_token_align']
            else:
                args.sub_token_align = False
                args.sub_factor = 1

            # 是否只为一半的层装备记忆力来节省显存消耗
            if config['model']['half_memory'] != 0:
                args.half_memory = True
            else:
                args.half_memory = False

            # 是否只有最后一层blk装备记忆力来节省显存消耗，避免记忆干扰当前帧的特征提取
            if config['model']['last_memory'] != 0:
                args.last_memory = True
            else:
                args.last_memory = False

            # 是否只有第一层blk装备记忆力
            if config['model']['early_memory'] != 0:
                args.early_memory = True
            else:
                args.early_memory = False

            # 是否只有中间blk装备记忆力
            config['model']['middle_memory'] = config['model'].get('middle_memory', 0)
            if config['model']['middle_memory'] != 0:
                args.middle_memory = True
            else:
                # default
                args.middle_memory = False

            # 是否使用cross attention融合记忆与当前特征(在Nh Nw维度流动信息)
            if config['model']['cross_att'] != 0:
                args.cross_att = True
            else:
                args.cross_att = False

            # 是否对时序上的信息也使用cross attention融合(额外在T维度流动信息)
            if config['model']['time_att'] != 0:
                args.time_att = True
            else:
                args.time_att = False

            # 是否在时序融合信息的时候解耦时空，降低计算复杂度
            if config['model']['time_deco'] != 0:
                args.time_deco = True
            else:
                args.time_deco = False

            # 是否在聚合时空记忆时使用temporal focal attention
            if config['model']['temp_focal'] != 0:
                args.temp_focal = True
            else:
                args.temp_focal = False

            # 是否在聚合时空记忆时使用cswin attention
            if config['model']['cs_win'] != 0:
                args.cs_win = True
                # if config['model']['cs_win'] == 2:
                #     # cs_win_strip决定了cswin的条带宽度，默认为1
                #     args.cs_win_strip = 2
                # else:
                #     args.cs_win_strip = 1
            else:
                args.cs_win = False
                # args.cs_win_strip = 1

            # 是否使用attention聚合不同时间的记忆和当前特征，而不是使用线性层聚合记忆再attention
            if config['model']['mem_att'] != 0:
                args.mem_att = True
            else:
                args.mem_att = False

            # 是否为cswin引入类似temporal focal的机制来增强注意力
            if config['model']['cs_focal'] != 0:
                args.cs_focal = True
                if config['model']['cs_focal'] == 2:
                    # 改进的正交全局滑窗策略，取到non-local的focal窗口
                    # 现在默认都是v2了，v1已经被淘汰
                    args.cs_focal_v2 = True
                else:
                    raise Exception('Focal v1 has been given up.')
            else:
                args.cs_focal = False
                args.cs_focal_v2 = False
        else:
            args.mem_pool = False
            args.store_lf = False
            args.align_cache = False
            args.sub_token_align = False
            args.sub_factor = 1
            args.early_memory = False
            args.half_memory = False
            args.last_memory = False
            args.middle_memory = False
            args.cross_att = False
            args.time_att = False
            args.time_deco = False
            args.temp_focal = False
            args.cs_win = False
            args.mem_att = False
            args.cs_focal = False
            args.cs_focal_v2 = False

        # 是否使用3D deco focav2 cswin替换temporal focal trans主干
        if config['model']['cs_trans'] != 0:
            args.cs_trans = True
        else:
            args.cs_trans = False

        # 是否使用MixF3N代替F3N，目前对两种transformer主干都生效
        if config['model']['mix_f3n'] != 0:
            args.mix_f3n = True
        else:
            args.mix_f3n = False

        # 是否使用FFN代替F3N，仅用于消融实验
        config['model']['ffn'] = config['model'].get('ffn', 0)
        if config['model']['ffn'] != 0:
            args.ffn = True
        else:
            # default
            args.ffn = False

        # 是否使用MixFFN代替F3N，仅用于消融实验，来自于SegFormer
        config['model']['mix_ffn'] = config['model'].get('mix_ffn', 0)
        if config['model']['mix_ffn'] != 0:
            args.mix_ffn = True
        else:
            # default
            args.mix_ffn = False

        # 定义transformer的深度
        if config['model']['depths'] != 0:
            args.depths = config['model']['depths']
        else:
            # 使用网络默认的深度
            args.depths = None

        # 定义trans主干不同层的head数量
        if config['model']['head_list'] != 0:
            args.head_list = config['model']['head_list']
        else:
            # 使用网络默认的head数量，也就是每层4个
            args.head_list = []

        # 定义不同的stage拥有多少个block
        if config['model']['blk_list'] != 0:
            args.blk_list = config['model']['blk_list']
        else:
            # 使用网络默认的blk数量，也就是深度的数量
            args.blk_list = []

        # 定义trans block的dim
        if config['model']['hide_dim'] != 0:
            args.hide_dim = config['model']['hide_dim']
        else:
            # 使用网络默认的blk数量，也就是深度的数量
            args.hide_dim = None

        # 定义trans block的window个数(token除以window划分大小)
        config['model']['window_size'] = config['model'].get('window_size', 0)
        if config['model']['window_size'] != 0:
            args.window_size = config['model']['window_size']
        else:
            # 使用网络默认的window
            args.window_size = None

        # # 定义trans block的输出大小
        # if config['model']['output_size'] != 0:
        #     args.output_size = config['model']['output_size']
        # else:
        #     # 使用网络默认的output_size
        #     args.output_size = None

        # 定义是大模型还是小模型
        if config['model']['small_model'] != 0:
            args.small_model = True
        else:
            args.small_model = False

        # 是否冻结dcn参数
        config['model']['freeze_dcn'] = config['model'].get('freeze_dcn', 0)
        if config['model']['freeze_dcn'] != 0:
            args.freeze_dcn = True
        else:
            # default
            args.freeze_dcn = False

        if args.cs_trans:
            # cs trans 主干需要的参数

            # 是否给attention加一个CONV path，目前仅对cs win trans block生效
            if config['model']['conv_path'] != 0:
                args.conv_path = True
            else:
                args.conv_path = False

            # 是否使用滑窗逻辑强化cs win，只对于条带宽度不为1时生效
            # 顺便更改了条带宽度不为1的池化逻辑，直接池化到条带的宽度，提高数据利用率(原来补0)
            if config['model']['cs_sw'] != 0:
                args.cs_sw = True
            else:
                args.cs_sw = False

            # 是否为cswin引入不同宽度条带池化的机制来增强注意力，只对初始条带宽度1有效
            if config['model']['pool_strip'] != 0:
                args.pool_strip = True
                if config['model']['pool_strip'] == 1:
                    # 使用什么宽度的条带来池化增强当前窗口
                    args.pool_sw = 1
                elif config['model']['pool_strip'] == 2:
                    args.pool_sw = 2
                elif config['model']['pool_strip'] == 4:
                    args.pool_sw = 4
                else:
                    raise Exception('Not implement.')
            else:
                args.pool_strip = False
                args.pool_sw = 2

            # 定义新trans主干不同层的条带宽度
            if config['model']['sw_list'] != 0:
                args.sw_list = config['model']['sw_list']
            else:
                # 使用网络默认的深度
                args.sw_list = []
    # # # # pass config to args # # # #

    return args


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


# sample reference frames from the whole video with mem support
# 允许相同的局部帧和非局部帧id，保证时间维度的一致性，但是引入了冗余计算？
# TODO:Dubug这里
def get_ref_index_mem(length, neighbor_ids, same_id=False):
    """smae_id(bool): If True, allow same ref and local id as input."""
    ref_index = []
    for i in range(0, length, ref_length):
        if same_id:
            # 允许相同id
            ref_index.append(i)
        else:
            # 不允许相同的id，当出现相同id时找到最近的一个不同的i
            if i not in neighbor_ids:
                ref_index.append(i)
            else:
                lf_id_avg = sum(neighbor_ids)/len(neighbor_ids)     # 计算 local frame id 平均值
                for _iter in range(0, 100):
                    if i < (length - 1):
                        # 不能超过视频长度
                        if i == 0:
                            # # 第0帧的时候重复，直接取到最后一个 LF + 2
                            # i = neighbor_ids[-1] + 2
                            # ref_index.append(i)
                            # 第0帧的时候重复，直接取到下一个 NLF + 5   +5是为了防止和下一个重复的 nlf id 改的id重复
                            i = ref_length + neighbor_stride
                            ref_index.append(i)
                            break
                        elif i < lf_id_avg:
                            # 往前找不重复的参考帧, 防止都往一个方向找而重复
                            i -= 1
                        else:
                            # 往后找不重复的参考帧
                            i += 1
                    else:
                        # 超过了直接用最后一帧，然后退出
                        ref_index.append(i)
                        break

                    if i not in neighbor_ids:
                        ref_index.append(i)
                        break

    return ref_index


# sample reference frames from the remain frames with random behavior like trainning
def get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3, before_nlf=False):
    """

    Args:
        neighbor_ids:
        video_length:
        num_ref_frame:
        before_nlf: If True, only sample refer frames from the past.

    Returns:

    """
    if not before_nlf:
        # 从过去和未来采集非局部帧
        complete_idx_set = list(range(video_length))
    else:
        # 非局部帧只会从过去的视频帧中选取，不会使用未来的信息
        complete_idx_set = list(range(neighbor_ids[-1]))
    # complete_idx_set = list(range(video_length))

    remain_idx = list(set(complete_idx_set) - set(neighbor_ids))

    # 当只用过去的帧作为非局部帧时，可能会出现过去的帧数量少于非局部帧需求的问题，比如视频的一开始
    if before_nlf:
        if len(remain_idx) < num_ref_frame:
            # 则我们允许从局部帧中采样非局部帧 转换为set可以去除重复元素
            remain_idx = list(set(remain_idx + neighbor_ids))

    ref_index = sorted(random.sample(remain_idx, num_ref_frame))
    return ref_index


def main_worker(args):
    args = read_cfg(args=args)      # 读取网络的所有设置
    w = args.output_size[0]
    h = args.output_size[1]
    args.size = (w, h)

    # set up datasets and data loader
    # default result
    test_dataset = TestDataset(args)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    try:
        if args.model == 'lite-MFN' or args.model == 'large-MFN':
            model = net.InpaintGenerator(
                skip_dcn=args.skip_dcn, freeze_dcn=args.freeze_dcn,
                spy_net=args.spy_net, flow_guide=args.flow_guide, flow_res=args.flow_res,
                token_fusion=args.token_fusion, fusion_recurrent=args.fusion_recurrent,
                token_fusion_simple=args.token_fusion_simple, fusion_skip_connect=args.fusion_skip_connect,
                memory=args.memory, max_mem_len=args.max_mem_len,
                compression_factor=args.compression_factor, mem_pool=args.mem_pool,
                store_lf=args.store_lf, align_cache=args.align_cache, sub_token_align=args.sub_token_align,
                sub_factor=args.sub_factor, half_memory=args.half_memory, last_memory=args.last_memory,
                early_memory=args.early_memory, middle_memory=args.middle_memory,
                cross_att=args.cross_att, time_att=args.time_att, time_deco=args.time_deco,
                temp_focal=args.temp_focal, cs_win=args.cs_win, mem_att=args.mem_att, cs_focal=args.cs_focal,
                cs_focal_v2=args.cs_focal_v2,
                cs_trans=args.cs_trans, mix_f3n=args.mix_f3n, ffn=args.ffn, mix_ffn=args.mix_ffn,
                depths=args.depths, head_list=args.head_list,
                blk_list=args.blk_list, hide_dim=args.hide_dim,
                window_size=args.model_win_size, output_size=args.model_output_size, small_model=args.small_model).to(device)
        else:
            # 加载一些尺寸窗口设置
            model = net.InpaintGenerator(window_size=args.model_win_size, output_size=args.model_output_size).to(device)
    except:
        try:
            # 加载一些尺寸窗口设置,sttn和fuseformer不需要window_size参数
            model = net.InpaintGenerator(output_size=args.model_output_size).to(device)
        except:
            model = net.InpaintGenerator().to(device)
    if args.ckpt is not None:
        data = torch.load(args.ckpt, map_location=device)
        if (args.model == 'fuseformer') or (args.model == 'sttn'):
            # sttn和fuseformer的gen ckpt额外嵌套了一层netG
            try:
                model.load_state_dict(data['netG'])
            except:
                model.load_state_dict(data)
        else:
            model.load_state_dict(data)
        print(f'Loading from: {args.ckpt}')

    # 计算FLOPs
    if args.FLOPs:
        myflops, flops, params = get_flops(model)   # paper FLOPs
        print(myflops)

        # input_shape = [1, 8, 3, 240, 432]
        # input_shape = (8, 3, 240, 432)
        # flops, params = get_model_complexity_info(model, input_shape)

        print('#############FLOPs:'+str(flops))
        print('#############PARAMES:'+str(params))

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
    # default
    # result_path = os.path.join('results', f'{args.model}_{args.dataset}')
    if args.ckpt is not None:
        ckpt = args.ckpt.split('/')[-1]
    else:
        ckpt = 'random'
    if args.fov is not None:
        if args.reverse:
            result_path = os.path.join('results', f'{args.model}+_{ckpt}_{args.fov}_{args.dataset}')
        else:
            result_path = os.path.join('results', f'{args.model}_{ckpt}_{args.fov}_{args.dataset}')
    else:
        if args.reverse:
            result_path = os.path.join('results', f'{args.model}+_{ckpt}_{args.dataset}')
        else:
            result_path = os.path.join('results', f'{args.model}_{ckpt}_{args.dataset}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    eval_summary = open(
        os.path.join(result_path, f"{args.model}_{args.dataset}_metrics.txt"),
        "w")

    i3d_model = init_i3d_model()

    for index, items in enumerate(test_loader):

        if args.memory:
            # 进入新的视频时清空记忆缓存
            # TODO: 更正记忆缓存清除逻辑
            for blk in model.transformer:
                try:
                    blk.attn.m_k = []
                    blk.attn.m_v = []
                except:
                    pass

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
        # 当这个循环走完的时候，一段视频已经被补全了
        for f in range(0, video_length, neighbor_stride):
            if not args.memory_logic:
                # default id with different T
                if not args.recurrent:
                    neighbor_ids = [
                        i for i in range(max(0, f - neighbor_stride),
                                         min(video_length, f + neighbor_stride + 1))
                    ]   # neighbor_ids即为Local Frames, 局部帧
                else:
                    # 在recurrent模式下，每次局部窗口都为1
                    neighbor_ids = [f]
            else:
                if args.same_memory:
                    # 尽可能与e2fgvi的原测试逻辑一致
                    # 输入的时间维度T保持一致
                    if (f - neighbor_stride > 0) and (f + neighbor_stride + 1 < video_length):
                        # 视频首尾均不会越界，不需要补充额外帧
                        neighbor_ids = [
                            i for i in range(max(0, f - neighbor_stride),
                                             min(video_length, f + neighbor_stride + 1))
                        ]  # neighbor_ids即为Local Frames, 局部帧
                    else:
                        # 视频越界，补充额外帧保证记忆缓存的时间通道维度一致，后面也可以尝试放到trans里直接复制特征的时间维度
                        neighbor_ids = [
                            i for i in range(max(0, f - neighbor_stride),
                                             min(video_length, f + neighbor_stride + 1))
                        ]  # neighbor_ids即为Local Frames, 局部帧
                        repeat_num = (neighbor_stride * 2 + 1) - len(neighbor_ids)

                        # for ii in range(0, repeat_num):
                        #     # 复制最后一帧
                        #     neighbor_ids.append(neighbor_ids[-1])

                        lf_id_avg = sum(neighbor_ids) / len(neighbor_ids)  # 计算 local frame id 平均值
                        last_id = neighbor_ids[-1]
                        first_id = neighbor_ids[0]
                        for ii in range(0, repeat_num):
                            # 保证局部窗口的大小一致，防止缓存通道数变化
                            if lf_id_avg < (video_length // 2):
                                # # 前半段视频向后找局部id
                                # new_id = last_id + 1 + ii
                                # 前半段视频也向前找局部id，防止和下一个窗口的输入完全一样
                                new_id = video_length - 1 - ii
                            else:
                                # 后半段视频向前找局部id
                                new_id = first_id - 1 - ii
                            neighbor_ids.append(new_id)

                        neighbor_ids = sorted(neighbor_ids)    # 重新排序

                else:
                    # 与记忆力模型的训练逻辑一致
                    if not args.recurrent:
                        if video_length < (f + neighbor_stride):
                            neighbor_ids = [
                                i for i in range(f, video_length)
                            ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次，视频尾部可能不足5帧局部帧，复制最后一帧补全数量
                            for repeat_idx in range(0, neighbor_stride - len(neighbor_ids)):
                                neighbor_ids.append(neighbor_ids[-1])
                        else:
                            neighbor_ids = [
                                i for i in range(f, f + neighbor_stride)
                            ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次
                    else:
                        # 在recurrent模式下，每次局部窗口都为1
                        neighbor_ids = [f]

            if not args.memory_logic:
                # default test set, 局部帧与非局部帧不会输入同样id的帧
                ref_ids = get_ref_index(neighbor_ids, video_length)  # ref_ids即为Non-Local Frames, 非局部帧

                selected_imgs = frames[:1, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            else:
                # 为了保证时间维度一致, 允许输入相同id的帧
                if args.same_memory:
                    ref_ids = get_ref_index_mem(video_length, neighbor_ids, same_id=args.same_id)  # ref_ids即为Non-Local Frames, 非局部帧
                elif args.past_ref:
                    ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3, before_nlf=True)  # 只允许过去的参考帧
                else:
                    ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3)  # 与序列训练同样的非局部帧输入逻辑

                ref_ids = sorted(ref_ids)  # 重新排序
                selected_imgs_lf = frames[:1, neighbor_ids, :, :, :]
                selected_imgs_nlf = frames[:1, ref_ids, :, :, :]
                selected_imgs = torch.cat((selected_imgs_lf, selected_imgs_nlf), dim=1)
                selected_masks_lf = masks[:1, neighbor_ids, :, :, :]
                selected_masks_nlf = masks[:1, ref_ids, :, :, :]
                selected_masks = torch.cat((selected_masks_lf, selected_masks_nlf), dim=1)

            with torch.no_grad():
                masked_frames = selected_imgs * (1 - selected_masks)

                if args.timing:
                    torch.cuda.synchronize()
                    time_start = time.time()
                if args.model == 'fuseformer':
                    # fuseformer不需要输入局部帧id因为没有分开处理
                    pred_img = model(masked_frames)
                elif args.model == 'sttn':
                    # sttn的前向需要同时输入mask_frames和mask
                    pred_img = model(masked_frames, selected_masks)
                else:
                    pred_img, _ = model(masked_frames, len(neighbor_ids))   # forward里会输入局部帧数量来对两种数据分开处理

                # 水平与竖直翻转增强
                if args.reverse:
                    masked_frames_horizontal_aug = torch.from_numpy(masked_frames.cpu().numpy()[:, :, :, :, ::-1].copy()).cuda()
                    pred_img_horizontal_aug, _ = model(masked_frames_horizontal_aug, len(neighbor_ids))
                    pred_img_horizontal_aug = torch.from_numpy(pred_img_horizontal_aug.cpu().numpy()[:, :, :, ::-1].copy()).cuda()
                    masked_frames_vertical_aug = torch.from_numpy(masked_frames.cpu().numpy()[:, :, :, ::-1, :].copy()).cuda()
                    pred_img_vertical_aug, _ = model(masked_frames_vertical_aug, len(neighbor_ids))
                    pred_img_vertical_aug = torch.from_numpy(pred_img_vertical_aug.cpu().numpy()[:, :, ::-1, :].copy()).cuda()

                    pred_img = 1 / 3 * (pred_img + pred_img_horizontal_aug + pred_img_vertical_aug)

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
                            # good_fusion下所有img多出一个‘次数’通道，用来记录所有的结果
                            comp_frames[idx] = img[np.newaxis, :, :, :]

                        # 针对默认滑窗的good fusion策略
                        # elif idx == (neighbor_ids[0] + neighbor_ids[-1])/2:
                        #     # 如果是中间帧，记录下来
                        #     medium_frame = img
                        # elif (idx != 0) & (idx == neighbor_ids[0]):
                        #     # 如果是第三次出现，加权平均
                        #     comp_frames[idx] = comp_frames[idx].astype(
                        #         np.float32) * 0.25 + medium_frame.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.25
                        # else:
                        #     # 如果是不是中间帧，权重为0.5
                        #     comp_frames[idx] = comp_frames[idx].astype(
                        #         np.float32) * 0.5 + img.astype(np.float32) * 0.5

                        # 直接把所有结果都记录下来，最后沿着通道平均
                        else:
                            comp_frames[idx] = np.concatenate((comp_frames[idx], img[np.newaxis, :, :, :]), axis=0)
                        ########################################################################################

        # 对于good_fusion, 推理一遍后需要沿着axis=0取平均
        if args.good_fusion:
            for idx, comp_frame in zip(range(0, video_length), comp_frames):
                comp_frame = comp_frame.astype(np.float32).sum(axis=0)/comp_frame.shape[0]
                comp_frames[idx] = comp_frame

        # 推理一遍后，额外的推理来刷记忆模型精度
        # TODO: 让这些额外推理与past_ref兼容
        if args.memory_double:
            for f in range(neighbor_stride//2, video_length, neighbor_stride):
                if not args.memory_logic:
                    # default id with different T
                    neighbor_ids = [
                        i for i in range(max(neighbor_stride//2, f - neighbor_stride),
                                         min(video_length, f + neighbor_stride + 1))
                    ]  # neighbor_ids即为Local Frames, 局部帧
                else:
                    if args.same_memory:
                        # 尽可能与e2fgvi的原测试逻辑一致
                        # 输入的时间维度T保持一致
                        if (f - neighbor_stride > 0) and (f + neighbor_stride + 1 < video_length):
                            # 视频首尾均不会越界，不需要补充额外帧
                            neighbor_ids = [
                                i for i in range(max(0, f - neighbor_stride),
                                                 min(video_length, f + neighbor_stride + 1))
                            ]   # neighbor_ids即为Local Frames, 局部帧
                        else:
                            # 视频越界，补充额外帧保证记忆缓存的时间通道维度一致，后面也可以尝试放到trans里直接复制特征的时间维度
                            neighbor_ids = [
                                i for i in range(max(0, f - neighbor_stride),
                                                 min(video_length, f + neighbor_stride + 1))
                            ]  # neighbor_ids即为Local Frames, 局部帧
                            repeat_num = (neighbor_stride * 2 + 1) - len(neighbor_ids)
                            for ii in range(0, repeat_num):
                                # 复制最后一帧
                                neighbor_ids.append(neighbor_ids[-1])

                    else:
                        # 与记忆力模型的训练逻辑一致
                        if video_length < (f + neighbor_stride):
                            neighbor_ids = [
                                i for i in range(f, video_length)
                            ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次，视频尾部可能不足5帧局部帧，复制最后一帧补全数量
                            for repeat_idx in range(0, neighbor_stride - len(neighbor_ids)):
                                neighbor_ids.append(neighbor_ids[-1])
                        else:
                            neighbor_ids = [
                                i for i in range(f, f + neighbor_stride)
                            ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次

                if not args.memory_logic:
                    # default test set, 局部帧与非局部帧不会输入同样id的帧
                    ref_ids = get_ref_index(neighbor_ids, video_length)  # ref_ids即为Non-Local Frames, 非局部帧

                    selected_imgs = frames[:1, neighbor_ids + ref_ids, :, :, :]
                    selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
                else:
                    # 为了保证时间维度一致, 允许输入相同id的帧
                    if args.same_memory:
                        ref_ids = get_ref_index_mem(video_length, neighbor_ids, same_id=args.same_id)  # ref_ids即为Non-Local Frames, 非局部帧
                    else:
                        ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3)  # 与序列训练同样的非局部帧输入逻辑

                    selected_imgs_lf = frames[:1, neighbor_ids, :, :, :]
                    selected_imgs_nlf = frames[:1, ref_ids, :, :, :]
                    selected_imgs = torch.cat((selected_imgs_lf, selected_imgs_nlf), dim=1)
                    selected_masks_lf = masks[:1, neighbor_ids, :, :, :]
                    selected_masks_nlf = masks[:1, ref_ids, :, :, :]
                    selected_masks = torch.cat((selected_masks_lf, selected_masks_nlf), dim=1)

                with torch.no_grad():
                    masked_frames = selected_imgs * (1 - selected_masks)

                    if args.timing:
                        torch.cuda.synchronize()
                        time_start = time.time()
                    pred_img, _ = model(masked_frames, len(neighbor_ids))  # forward里会输入局部帧数量来对两种数据分开处理
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

                            else:  # default 融合策略：不合理，neighbor_stride倍数的LF的中间帧权重为0.25，应当为0.5
                                # 如果不是第一次补全Local Frame中的某帧，即该帧已补全过，则把此前结果与当前帧结果简单加和平均
                                comp_frames[idx] = comp_frames[idx].astype(
                                    np.float32) * 0.5 + img.astype(np.float32) * 0.5
                            ########################################################################################
                        else:
                            if comp_frames[idx] is None:
                                # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                                comp_frames[idx] = img

                            elif idx == (neighbor_ids[0] + neighbor_ids[-1]) / 2:
                                # 如果是中间帧，记录下来
                                medium_frame = img
                            elif (idx != 0) & (idx == neighbor_ids[0]):
                                # 如果是第三次出现，加权平均
                                comp_frames[idx] = comp_frames[idx].astype(
                                    np.float32) * 0.25 + medium_frame.astype(np.float32) * 0.5 + img.astype(
                                    np.float32) * 0.25
                            else:
                                # 如果是不是中间帧，权重为0.5
                                comp_frames[idx] = comp_frames[idx].astype(
                                    np.float32) * 0.5 + img.astype(np.float32) * 0.5
                            ########################################################################################
        elif args.memory_fifth:
            # 丧心病狂推理5次来刷精度
            for sliding_start in range(1, neighbor_stride):
                for f in range(sliding_start, video_length, neighbor_stride):
                    if not args.memory_logic:
                        raise Exception('Not support aug with no memory models')
                    else:
                        if args.same_memory:
                            raise Exception('Not support aug with same behavior of e2fgvi')
                        else:
                            # 与记忆力模型的训练逻辑一致
                            if video_length < (f + neighbor_stride):
                                neighbor_ids = [
                                    i for i in range(f, video_length)
                                ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次，视频尾部可能不足5帧局部帧，复制最后一帧补全数量
                                for repeat_idx in range(0, neighbor_stride - len(neighbor_ids)):
                                    neighbor_ids.append(neighbor_ids[-1])
                            else:
                                neighbor_ids = [
                                    i for i in range(f, f + neighbor_stride)
                                ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次

                    if not args.memory_logic:
                        raise Exception('Not support aug with no memory models')
                    else:
                        # 为了保证时间维度一致, 允许输入相同id的帧
                        if args.same_memory:
                            raise Exception('Not support aug with same behavior of e2fgvi')
                        else:
                            ref_ids = get_ref_index_mem_random(neighbor_ids, video_length,
                                                               num_ref_frame=3)  # 与序列训练同样的非局部帧输入逻辑

                        selected_imgs_lf = frames[:1, neighbor_ids, :, :, :]
                        selected_imgs_nlf = frames[:1, ref_ids, :, :, :]
                        selected_imgs = torch.cat((selected_imgs_lf, selected_imgs_nlf), dim=1)
                        selected_masks_lf = masks[:1, neighbor_ids, :, :, :]
                        selected_masks_nlf = masks[:1, ref_ids, :, :, :]
                        selected_masks = torch.cat((selected_masks_lf, selected_masks_nlf), dim=1)

                    with torch.no_grad():
                        masked_frames = selected_imgs * (1 - selected_masks)

                        if args.timing:
                            torch.cuda.synchronize()
                            time_start = time.time()
                        pred_img, _ = model(masked_frames, len(neighbor_ids))  # forward里会输入局部帧数量来对两种数据分开处理
                        if args.timing:
                            torch.cuda.synchronize()
                            time_end = time.time()
                            time_sum = time_end - time_start
                            time_all += time_sum

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

                                else:  # default 融合策略：不合理，neighbor_stride倍数的LF的中间帧权重为0.25，应当为0.5
                                    # 如果不是第一次补全Local Frame中的某帧，即该帧已补全过，则把此前结果与当前帧结果简单加和平均
                                    comp_frames[idx] = comp_frames[idx].astype(
                                        np.float32) * 0.5 + img.astype(np.float32) * 0.5
                            else:
                                raise Exception('Not support aug with good fusion.')

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
    parser = argparse.ArgumentParser(description='FlowLens')
    parser.add_argument('--cfg_path', default='configs/KITTI360EX-I_FlowLens_early_small_v2.json')
    parser.add_argument('--dataset',
                        choices=['davis', 'youtube-vos', 'pal', 'KITTI360-EX'],
                        type=str)       # 相当于train的‘name’
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--output_size', type=int, nargs='+', default=[432, 240])
    parser.add_argument('--object', action='store_true', default=False)     # if true, use object removal mask
    parser.add_argument('--fov',
                        choices=['fov5', 'fov10', 'fov20'],
                        type=str)  # 对于KITTI360-EX, 测试需要输入fov
    parser.add_argument('--past_ref', action='store_true', default=False)  # 对于KITTI360-EX, 测试时只允许使用之前的参考帧
    parser.add_argument('--model', choices=[
        'e2fgvi', 'e2fgvi_hq', 'e2fgvi_hq-lite', 'lite-MFN', 'large-MFN', 'fuseformer', 'sttn'], type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--timing', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--good_fusion', action='store_true', default=False, help='using my fusion strategy')
    parser.add_argument('--memory_logic', action='store_true', default=False,
                        help='test with memory logic, support for VI-Trans and FlowLens')
    parser.add_argument('--same_memory', action='store_true', default=False,
                        help='test with memory ability in E2FGVI style, not work with --memory_double')
    parser.add_argument('--same_id', action='store_true', default=False,
                        help='if True, allow same ref id and local id input for memory models.')
    # TODO: 这里的memory double逻辑还可以把前面两帧也再次估计一遍提升精度
    parser.add_argument('--memory_double', action='store_true', default=False, help='test with memory ability twice')
    parser.add_argument('--memory_fifth', action='store_true', default=False, help='test with memory ability five times')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='test with horizontal and vertical reverse augmentation')
    parser.add_argument('--model_win_size', type=int, nargs='+', default=[5, 9])
    parser.add_argument('--model_output_size', type=int, nargs='+', default=[60, 108])
    parser.add_argument('--FLOPs', action='store_true', default=False,
                        help='calc FLOPs of the model')
    parser.add_argument('--recurrent', action='store_true', default=False,
                        help='keep window = 1, stride = 1 to not use future info')
    args = parser.parse_args()

    if args.dataset == 'KITTI360-EX':
        # 对于KITTI360-EX, 测试时只允许使用之前的参考帧
        args.past_ref = True

    if args.profile:
        # profile = line_profiler.LineProfiler(main_worker)  # 把函数传递到性能分析器
        # profile.enable()  # 开始分析
        pass

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
        # profile.disable()  # 停止分析
        # profile.print_stats(sys.stdout)  # 打印出性能分析结果
        pass
