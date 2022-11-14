import cv2
import numpy as np
import argparse
import os


def folder_name(file_dir):
    """获取当前目录下的目录列表"""
    dir_list = []
    for root, dirs, files in os.walk(file_dir):
        dir_list.append(dirs)
    return dir_list[0]


def file_name(file_dir, suffix='.png'):
    """获取当前目录下后缀为suffix的文件列表"""
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L


parser = argparse.ArgumentParser()
parser.add_argument('--copy_number', default=10, type=int)
parser.add_argument('--src_mask_path', default='H://davis//test_masks_object//', type=str)
args = parser.parse_args()

# 获取语义mask的所有文件夹
video_list = folder_name(args.src_mask_path)

# 视频工作循环
for video in video_list:

    # 当前video的path
    video_path = os.path.join(args.src_mask_path, video)

    # 获取当前video下的mask列表
    mask_list = file_name(video_path)

    # Mask工作循环
    idx = 0     # Mask的id
    for mask_path in mask_list:
        mask = cv2.imread(mask_path)
        valid_mask = (((mask[:, :, 0] != 0) + (mask[:, :, 1] != 0) + (mask[:, :, 2] != 0))*255).astype(np.uint8)
        mask = cv2.resize(valid_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(video_path, str(idx).zfill(5) + '.png'), mask)
        print('Generate Object Mask: %d' % idx)
        idx += 1
