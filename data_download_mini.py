#!/usr/bin/env python3
import os
import random
import pandas as pd
import numpy as np
import torchaudio
import librosa
from tqdm import tqdm

def download_test_clean(root_dir):
    print("Downloading test-clean subset...")
    torchaudio.datasets.LIBRISPEECH(root=root_dir, url="test-clean", download=True)
    print("Download complete.\n")

def build_mini_metadata(root_dir, out_csv, max_per_speaker=5):
    """
    构建小规模元数据，每个 speaker 取最多 N 条音频
    """
    subset_dir = os.path.join(root_dir, "LibriSpeech", "test-clean")
    records = []

    for speaker in os.listdir(subset_dir):
        spk_dir = os.path.join(subset_dir, speaker)
        for chapter in os.listdir(spk_dir):
            ch_dir = os.path.join(spk_dir, chapter)
            for fname in os.listdir(ch_dir):
                if fname.endswith(".flac"):
                    records.append({
                        "speaker_id": int(speaker),
                        "file_path": os.path.join(ch_dir, fname)
                    })

    df = pd.DataFrame(records)
    # 只保留每个 speaker 最多 N 条
    df = df.groupby("speaker_id").apply(lambda x: x.sample(min(max_per_speaker, len(x)))).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Mini metadata saved to {out_csv}, 共 {len(df)} 条音频。\n")
    return df

def extract_melspec(df, out_dir, sr=16000, n_mels=64):
    os.makedirs(out_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting melspec"):
        y, _ = librosa.load(row.file_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        base = os.path.basename(row.file_path).replace(".flac", "")
        np.save(os.path.join(out_dir, f"{row.speaker_id}_{base}.npy"), log_mel)
    print(f"Features saved to {out_dir}\n")

def main():
    root_dir = "./data"
    meta_csv = "metadata_mini.csv"
    feat_dir = "melspec_mini"

    os.makedirs(root_dir, exist_ok=True)
    download_test_clean(root_dir)
    df = build_mini_metadata(root_dir, meta_csv)
    extract_melspec(df, feat_dir)

if __name__ == "__main__":
    main()
