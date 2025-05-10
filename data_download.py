#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import torchaudio
import librosa
from tqdm import tqdm

def download_subsets(root_dir, subsets):
    os.makedirs(root_dir, exist_ok=True)  # <== 添加这行
    for subset in subsets:
        print(f"Downloading {subset} ...")
        torchaudio.datasets.LIBRISPEECH(
            root=root_dir,
            url=subset,
            download=True
        )
    print("All subsets downloaded.\n")

def build_metadata(root_dir, subsets, out_csv):
    """
    遍历数据目录，生成 metadata.csv
    columns: [file_path, speaker_id]
    """
    records = []
    for subset in subsets:
        subset_dir = os.path.join(root_dir, "LibriSpeech", subset)
        for speaker in os.listdir(subset_dir):
            spk_dir = os.path.join(subset_dir, speaker)
            if not os.path.isdir(spk_dir): continue
            for chapter in os.listdir(spk_dir):
                ch_dir = os.path.join(spk_dir, chapter)
                for fname in os.listdir(ch_dir):
                    if fname.endswith(".flac") or fname.endswith(".wav"):
                        path = os.path.join(ch_dir, fname)
                        records.append({
                            "file_path": path,
                            "speaker_id": int(speaker)
                        })
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Metadata saved to {out_csv}  共 {len(df)} 条记录。\n")
    return df

def extract_melspec(df, out_dir, sr=16000, n_mels=64):
    """
    对每条音频提取梅尔谱，并保存为 .npy
    文件名格式：<speaker_id>_<原文件名>.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting melspec"):
        wav_path = row.file_path
        spk = row.speaker_id
        y, _ = librosa.load(wav_path, sr=sr)
        # 计算梅尔谱 (shape: n_mels x T)
        mels = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        # 对数变换
        log_mels = librosa.power_to_db(mels, ref=np.max)
        base = os.path.basename(wav_path).replace(".flac", "").replace(".wav", "")
        out_path = os.path.join(out_dir, f"{spk}_{base}.npy")
        np.save(out_path, log_mels)
    print(f"All features extracted to {out_dir}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Download & preprocess LibriSpeech for speaker recognition"
    )
    parser.add_argument(
        "--root", type=str, default="./data",
        help="数据下载和存放的根目录"
    )
    parser.add_argument(
        "--subsets", nargs="+",
        default=[
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "test-clean"
        ],
        help="要下载并处理的 LibriSpeech 子集"
    )
    parser.add_argument(
        "--meta_csv", type=str, default="metadata.csv",
        help="输出 metadata CSV 文件名"
    )
    parser.add_argument(
        "--feat_dir", type=str, default="melspec",
        help="输出梅尔谱特征的目录名"
    )
    args = parser.parse_args()

    # 1. 下载
    download_subsets(args.root, args.subsets)

    # 2. 构建元数据
    df = build_metadata(args.root, args.subsets, args.meta_csv)

    # 3. 提取梅尔谱特征
    extract_melspec(df, args.feat_dir)

if __name__ == "__main__":
    main()
