"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
from tqdm import tqdm
import pathlib
import openslide
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

def load_svs_image(svs_path, level=0):
    """
    讀取 SVS 檔案，返回指定層級的影像 NumPy 陣列。
    :param svs_path: SVS 檔案路徑
    :param level: 要讀取的層次（0 表示最高分辨率）
    :return: NumPy 陣列格式的影像
    """
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.level_dimensions[level]
    pil_image = slide.read_region((0, 0), level, (width, height)).convert("RGB")
    return np.array(pil_image)

def extract_patches(image, win_size, step_size):
    """
    根據指定的窗口大小和步長，從影像中切出所有補丁。
    :param image: 輸入的 NumPy 影像陣列
    :param win_size: 補丁大小 [高度, 寬度]
    :param step_size: 步長大小 [高度, 寬度]
    :return: 補丁列表
    """
    h, w, c = image.shape
    patch_height, patch_width = win_size
    step_height, step_width = step_size

    patches = []
    print("Extracting patches...")
    for y in tqdm(range(0, h - patch_height + 1, step_height), desc="Rows", position=0):
        for x in tqdm(range(0, w - patch_width + 1, step_width), desc="Cols", position=1, leave=False):
            patch = image[y:y+patch_height, x:x+patch_width, :]
            patches.append(patch)
    return patches

def save_patches_as_npy(patches, save_dir, base_name):
    """
    將補丁保存為 .npy 檔案。
    :param patches: 補丁列表
    :param save_dir: 保存目錄
    :param base_name: 檔案基本名稱
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Saving patches...")
    for idx, patch in tqdm(enumerate(patches), total=len(patches), desc="Saving Patches"):
        save_path = os.path.join(save_dir, f"{base_name}_{idx:03d}.npy")
        np.save(save_path, patch)
    print(f"Saved {len(patches)} patches to {save_dir}")

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    svs_path = "/data1/johnny99457/DATASETS/TCGA/TEST/TCGA-49-4505-01Z-00-DX4.623c4278-fc3e-4c80-bb4d-000e24fbb1c2.svs"  # 替換為您的 WSI 檔案路徑
    save_dir = "/data1/johnny99457/hover_net/dataset/TCGA-49-4505-01Z-00-DX4_patches_40x"  # 替換為保存補丁的目錄
    win_size = [270, 270]  # 補丁大小 (高度, 寬度)
    step_size = [270, 270]  # 步長 (高度, 寬度)
    level = 0  # WSI 層級（0 表示最高解析度）

    # 讀取影像
    print("Loading WSI image...")
    image = load_svs_image(svs_path, level=level)
    print(f"WSI image loaded with shape: {image.shape}")

    # 提取補丁
    patches = extract_patches(image, win_size, step_size)
    print(f"Extracted {len(patches)} patches.")

    # 保存補丁為 .npy
    base_name = os.path.splitext(os.path.basename(svs_path))[0]  # 獲取檔案名稱
    save_patches_as_npy(patches, save_dir, base_name)
