# data_download_mini.py
#!/usr/bin/env python3
import os
import random
import pandas as pd
import numpy as np
import torchaudio
import librosa
from tqdm import tqdm

from utils import download_librispeech_subset, build_mini_metadata, extract_melspec_from_file
# 配置：每个子集最多取 3 个说话人，每人 2 条
MAX_SPEAKERS = 3
MAX_PER_SPK = 2

def download_and_build(root_dir, subset, meta_csv):
    """使用 until.py 下载并构建小规模 metadata"""
    print(f"Downloading {subset}...")
    download_librispeech_subset(root_dir, subset)
    print(f"Building mini metadata for {subset}...")
    df = build_mini_metadata(root_dir, subset,
                              max_speakers=MAX_SPEAKERS,
                              max_per_spk=MAX_PER_SPK)
    df.to_csv(meta_csv, index=False)
    print(f"{subset} metadata saved to {meta_csv} ({len(df)} 条)\n")
    return df

def extract_melspec(df, out_dir, sr=16000, n_mels=64):
    """对 metadata 中的所有文件提取梅尔谱，按 subset 子目录存放"""
    for subset in df['subset'].unique():
        os.makedirs(os.path.join(out_dir, subset), exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting melspec"):
        log_mel = extract_melspec_from_file(row.file_path, sr=sr, n_mels=n_mels)
        base = os.path.basename(row.file_path).replace(".flac", "")
        out_path = os.path.join(out_dir, row.subset, f"{row.speaker_id}_{base}.npy")
        np.save(out_path, log_mel)
    print(f"All features saved to {out_dir}\n")


def main():
    root_dir = "./audio"
    os.makedirs(root_dir, exist_ok=True)

    # 1. 下载并构建元数据
    df_train = download_and_build(root_dir, "train-clean-100", "meta_train_small.csv")
    df_test  = download_and_build(root_dir, "test-clean",        "meta_test_small.csv")
    df_all   = pd.concat([df_train, df_test], ignore_index=True)
    df_all.to_csv("metadata_small.csv", index=False)
    print(f"Combined metadata saved to metadata_small.csv ({len(df_all)} files)\n")

    # 3. 提取梅尔谱特征
    extract_melspec(df_all, "melspec_small")

if __name__ == "__main__":
    main()