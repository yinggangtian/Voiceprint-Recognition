"""
通用工具模块：包含数据预处理、DataLoader、VAD、文件查找、检查点管理和可视化函数
"""
import os
import re
import logging
import random
from glob import glob

import numpy as np
import pandas as pd
import librosa
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from silence_detector import SilenceDetector
import constants as c


# ----------------------------------------
# 数据集和预处理函数
# ----------------------------------------

def download_librispeech_subset(root_dir, subset_url):
    """下载指定的 LibriSpeech 子集"""
    os.makedirs(root_dir, exist_ok=True)
    logging.info(f"Downloading {subset_url} subset to {root_dir}")
    torchaudio.datasets.LIBRISPEECH(root=root_dir, url=subset_url, download=True)

def build_mini_metadata(root_dir, subset_url, max_speakers=3, max_per_spk=2):
    """
    构建小规模 metadata：随机抽取说话人和样本
    返回 DataFrame with columns ['speaker_id','file_path','subset']
    """
    subset_dir = os.path.join(root_dir, "LibriSpeech", subset_url)
    speakers = [
        d for d in os.listdir(subset_dir)
        if os.path.isdir(os.path.join(subset_dir, d))
    ]
    chosen = random.sample(speakers, min(max_speakers, len(speakers)))

    records = []
    for spk in chosen:
        spk_dir = os.path.join(subset_dir, spk)
        flac_files = []
        for chap in os.listdir(spk_dir):
            chap_dir = os.path.join(spk_dir, chap)
            flac_files += [
                os.path.join(chap_dir, f)
                for f in os.listdir(chap_dir)
                if f.endswith('.flac')
            ]
        sampled = random.sample(flac_files, min(len(flac_files), max_per_spk))
        for path in sampled:
            records.append({
                'speaker_id': int(spk),
                'file_path' : path,
                'subset'    : subset_url
            })

    return pd.DataFrame(records)


def load_metadata(csv_path):
    """加载 metadata CSV"""
    return pd.read_csv(csv_path)


def split_metadata(df, train_frac=0.8, test_size=None, shuffle=True, seed=42):
    """
    划分 train/test metadata
    
    参数:
        df: DataFrame 包含数据的元信息
        train_frac: 训练集比例 (与 test_size 二选一)
        test_size: 测试集比例 (与 train_frac 二选一)
        shuffle: 是否打乱数据
        seed: 随机种子
        
    返回:
        train_df, test_df: 训练集和测试集的 DataFrame
    """
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    if test_size is not None:
        # 使用 test_size
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    else:
        # 使用 train_frac
        idx = int(len(df) * train_frac)
        train_df, test_df = df.iloc[:idx], df.iloc[idx:]
    
    return train_df, test_df


def split_data(files, labels, test_size=0.2, random_state=42):
    """
    划分训练集和测试集 (列表形式)
    
    参数:
        files: 文件路径列表
        labels: 标签列表
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        train_files, test_files, train_labels, test_labels
    """
    return train_test_split(files, labels, test_size=test_size, random_state=random_state)


# ----------------------------------------
# 特征提取函数
# ----------------------------------------

def extract_melspec(audio, sr=16000, n_mels=64):
    """
    从音频数据提取梅尔频谱特征
    
    参数:
        audio: 音频时间序列数据
        sr: 采样率
        n_mels: 梅尔频谱的频带数量
        
    返回:
        log_mel: 对数梅尔频谱特征
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def extract_melspec_from_file(file_path, sr=16000, n_mels=64):
    """
    从音频文件提取梅尔频谱特征
    
    参数:
        file_path: 音频文件路径
        sr: 采样率
        n_mels: 梅尔频谱的频带数量
        
    返回:
        log_mel: 梅尔频谱特征
    """
    y, _ = librosa.load(file_path, sr=sr)
    return extract_melspec(y, sr, n_mels)


def extract_and_save_melspec(df, out_dir, sr=16000, n_mels=64):
    """
    提取梅尔谱并保存为 .npy
    按 subset 子目录组织
    
    参数:
        df: DataFrame 包含音频文件路径信息
        out_dir: 输出目录
        sr: 采样率
        n_mels: 梅尔频谱的频带数量
    """
    # 创建子集目录
    for subset in df['subset'].unique():
        os.makedirs(os.path.join(out_dir, subset), exist_ok=True)
    
    # 处理每个音频文件
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting melspec'):
        # 加载音频
        y, _ = librosa.load(row['file_path'], sr=sr)
        
        # 提取特征
        log_mel = extract_melspec(y, sr, n_mels)
        
        # 保存特征
        base = os.path.basename(row['file_path']).replace('.flac', '')
        save_path = os.path.join(out_dir, row['subset'], f"{row['speaker_id']}_{base}.npy")
        np.save(save_path, log_mel)


# ----------------------------------------
# 标签映射函数
# ----------------------------------------

def build_label_map(df, speaker_col='speaker_id'):
    """
    从 metadata 构建 label-to-id 映射
    
    参数:
        df: DataFrame 包含说话人标签
        speaker_col: 包含说话人ID的列名
        
    返回:
        label_map: 标签到索引的字典映射 {speaker_id: index}
    """
    labels = np.unique(df[speaker_col])
    return {label: idx for idx, label in enumerate(labels)}


# ----------------------------------------
# 数据加载函数
# ----------------------------------------

def load_features_and_labels(paths, label_map, no_of_speakers):
    """
    从 .npy 加载特征，生成特征和标签(one-hot)
    
    参数:
        paths: npy文件路径列表
        label_map: 标签到索引的映射
        no_of_speakers: 说话人总数
        
    返回:
        x: 特征数组
        y: one-hot编码的标签
    """
    feats, targets = [], []
    expected_shape = None
    
    for p in paths:
        if not os.path.exists(p):
            continue
            
        # 加载特征
        arr = np.load(p)
        
        # 检查特征形状一致性
        if expected_shape is None:
            expected_shape = arr.shape
        if arr.shape != expected_shape:
            logging.warning(f"形状不匹配 {p}: {arr.shape} != {expected_shape}")
            continue
            
        # 提取说话人ID并映射
        speaker = int(os.path.basename(p).split('_')[0])
        if speaker not in label_map:
            logging.warning(f"未映射说话人 {speaker}")
            continue
            
        feats.append(arr)
        targets.append(label_map[speaker])
        
    if not feats:
        raise RuntimeError("没有有效特征")
        
    x = np.stack(feats)
    y = np.eye(no_of_speakers)[targets]  # one-hot编码
    return x, y


def batch_data_loader(data_paths, label_map, no_of_speakers, batch_size):
    """
    无限循环生成批次数据的生成器
    
    参数:
        data_paths: 数据路径列表
        label_map: 标签到索引的映射
        no_of_speakers: 说话人总数
        batch_size: 批次大小
        
    yields:
        x_batch, y_batch: 特征批次和对应标签
    """
    while True:
        random.shuffle(data_paths)
        for i in range(0, len(data_paths), batch_size):
            batch = data_paths[i:i+batch_size]
            yield load_features_and_labels(batch, label_map, no_of_speakers)


def paths_to_loaders(meta_df, feature_root, label_map, batch_size, loader_fn=batch_data_loader):
    """
    根据 metadata DF 生成 DataLoader
    
    参数:
        meta_df: 元数据DataFrame
        feature_root: 特征文件根目录
        label_map: 标签映射
        batch_size: 批次大小
        loader_fn: 加载器函数
        
    返回:
        data_loader: 数据加载器
    """
    files = []
    labels = []
    
    for _, row in meta_df.iterrows():
        subset = row['subset']
        spk = row['speaker_id']
        base = os.path.basename(row['file_path']).replace('.flac', '')
        fname = os.path.join(feature_root, subset, f"{spk}_{base}.npy")
        files.append(fname)
        labels.append(label_map[spk])
        
    return loader_fn(files, label_map, len(label_map), batch_size)


# ----------------------------------------
# VAD（语音活动检测）函数
# ----------------------------------------

def vad_trim(audio, sr, silence_threshold_db=20):
    """
    基于VAD裁剪音频，移除静音段
    
    参数:
        audio: 音频数据
        sr: 采样率
        silence_threshold_db: 静音阈值(dB)
        
    返回:
        trimmed_audio: 裁剪后的音频
    """
    chunk_size = int(sr * 0.05)  # 50ms 分块
    sd = SilenceDetector(silence_threshold_db)
    output = []
    
    # 遍历每个分块
    for i in range(0, len(audio), chunk_size):
        chunk_data = audio[i:i+chunk_size]
        if not sd.is_silence(chunk_data):
            output.extend(chunk_data)
            
    return np.array(output)


# ----------------------------------------
# 文件查找和排序函数
# ----------------------------------------

def find_files(directory, pattern='**/*.wav'):
    """
    递归查找匹配文件模式的所有文件
    
    参数:
        directory: 根目录
        pattern: 文件匹配模式
        
    返回:
        files: 匹配的文件路径列表
    """
    return glob(os.path.join(directory, pattern), recursive=True)


def natural_sort(file_list):
    """
    自然排序字符串列表 (处理数字部分)
    
    参数:
        file_list: 要排序的字符串列表
        
    返回:
        sorted_list: 自然排序后的列表
    """
    convert = lambda t: int(t) if t.isdigit() else t.lower()
    return sorted(file_list, key=lambda fn: [convert(c) for c in re.split('([0-9]+)', fn)])


# ----------------------------------------
# 检查点(checkpoint)管理函数
# ----------------------------------------

def get_last_checkpoint(checkpoint_dir):
    """
    获取最新的检查点文件
    
    参数:
        checkpoint_dir: 检查点目录
        
    返回:
        最新的检查点文件路径，如果没有则返回 None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 同时识别 .h5 和 .weights.h5 扩展名，以兼容新旧格式
    files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) 
             if f.endswith('.h5') or f.endswith('.weights.h5')]
    if not files:
        return None
    return natural_sort(files)[-1]


def clean_old_checkpoints(checkpoint_dir, keep_latest=4):
    """
    清理旧的检查点文件，仅保留最新的几个
    
    参数:
        checkpoint_dir: 检查点目录
        keep_latest: 保留的最新检查点数量
    """
    # 同时识别 .h5 和 .weights.h5 扩展名，以兼容新旧格式
    files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) 
             if f.endswith('.h5') or f.endswith('.weights.h5')]
    sorted_files = sorted(files, key=os.path.getmtime)
    
    # 删除旧的检查点文件
    for old in sorted_files[:-keep_latest]:
        logging.info(f"删除旧模型: {old}")
        os.remove(old)


# ----------------------------------------
# 可视化函数
# ----------------------------------------

def plot_metrics(log_file, columns, out_path, labels=None):
    """
    通用绘制日志曲线函数
    
    参数:
        log_file: 日志文件路径
        columns: 要绘制的列
        out_path: 图表保存路径
        labels: 图例标签，默认使用列名
    """
    data = pd.read_csv(log_file, names=['step'] + columns)
    plt.figure(figsize=(10, 6))
    
    if labels is None:
        labels = columns
        
    for col, lab in zip(columns, labels):
        plt.plot(data['step'], data[col], label=lab)
        
    plt.legend()
    plt.xlabel('Steps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_loss_acc(file_path, out_path='figs/loss_acc.png'):
    """
    绘制损失和准确率曲线
    
    参数:
        file_path: 日志文件路径
        out_path: 图表保存路径
    """
    # 加载数据
    step = []
    loss = []
    acc = []
    mov_loss = []
    mov_acc = []
    ml, mv = 0, 0
    
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:  # 至少要有 step, loss, acc
                step.append(int(parts[0]))
                loss.append(float(parts[1]))
                acc.append(float(parts[-1]))
                
                # 计算移动平均
                if ml == 0:
                    ml = float(parts[1])
                    mv = float(parts[-1])
                else:
                    ml = 0.01 * float(parts[1]) + 0.99 * mov_loss[-1]
                    mv = 0.01 * float(parts[-1]) + 0.99 * mov_acc[-1]
                    
                mov_loss.append(ml)
                mov_acc.append(mv)
    
    # 创建子图
    plt.figure(figsize=(10, 10))
    
    # 绘制损失曲线
    plt.subplot(211)
    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels=['loss', 'moving_average_loss'], loc='best')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(212)
    p1, = plt.plot(step, acc)
    p2, = plt.plot(step, mov_acc)
    plt.legend(handles=[p1, p2], labels=['Accuracy', 'moving_average_accuracy'], loc='best')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_speaker_metrics(file_path, out_path='figs/acc_eer.png'):
    """
    绘制说话人识别相关指标 (EER, F-measure, Accuracy)
    
    参数:
        file_path: 日志文件路径
        out_path: 图表保存路径
    """
    # 加载数据
    step = []
    eer = []
    fm = []  # F-measure
    acc = []
    mov_eer = []
    mv = 0
    
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:  # 需要 step, eer, fm, acc
                step.append(int(parts[0]))
                eer.append(float(parts[1]))
                fm.append(float(parts[2]))
                acc.append(float(parts[3]))
                
                # 计算移动平均 EER
                if mv == 0:
                    mv = float(parts[1])
                else:
                    mv = 0.1 * float(parts[1]) + 0.9 * mov_eer[-1]
                    
                mov_eer.append(mv)
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    p1, = plt.plot(step, fm, color='black')
    p2, = plt.plot(step, eer, color='blue')
    p3, = plt.plot(step, acc, color='red')
    p4, = plt.plot(step, mov_eer, color='green')
    
    plt.xlabel("Steps")
    plt.grid(True, alpha=0.3)
    plt.legend(handles=[p1, p2, p3, p4], labels=['F-measure', 'EER', 'Accuracy', 'Moving_Average_EER'], loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


# ----------------------------------------
# 辅助工具函数
# ----------------------------------------

def change_filename(path, separator='_', new_separator='-'):
    """
    批量更改文件名，重新格式化说话人ID和话语ID
    
    参数:
        path: 文件所在目录
        separator: 原分隔符
        new_separator: 新分隔符
    """
    files = os.listdir(path)
    for file in files:
        # 替换破折号为下划线
        name = file.replace('-', '_')
        # 分割
        lis = name.split('_')
        # 合并说话人ID部分和话语ID部分
        speaker = '_'.join(lis[:3])
        utt_id = '_'.join(lis[3:])
        newname = speaker + new_separator + utt_id
        # 重命名
        os.rename(os.path.join(path, file), os.path.join(path, newname))


def copy_wav(kaldi_dir, out_dir):
    """
    从Kaldi目录结构复制WAV文件并重命名
    
    参数:
        kaldi_dir: Kaldi数据目录
        out_dir: 输出目录
    """
    import shutil
    from time import time
    
    orig_time = time()
    
    # 读取utt2spk映射
    with open(os.path.join(kaldi_dir, 'utt2spk'), 'r') as f:
        utt2spk = f.readlines()
    
    # 读取wav.scp映射
    with open(os.path.join(kaldi_dir, 'wav.scp'), 'r') as f:
        wav2path = f.readlines()
    
    # 构建话语到路径的映射
    utt2path = {}
    for wav in wav2path:
        parts = wav.split()
        if len(parts) >= 2:
            utt, path = parts[0], parts[1]
            utt2path[utt] = path
    
    logging.info(f"开始复制 {len(utt2path)} 个音频文件到 {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 复制并重命名文件
    for i, line in enumerate(utt2spk):
        parts = line.split()
        if len(parts) < 2:
            continue
            
        utt_id_parts = parts[0].split('_')[:-1]  # 去除最后一个部分
        utt_id = '_'.join(utt_id_parts)
        speaker = parts[1]
        
        # 检查路径是否存在
        if utt_id not in utt2path:
            logging.warning(f"未找到话语ID {utt_id} 的路径")
            continue
            
        filepath = utt2path[utt_id]
        
        # 重命名以符合 LibriSpeech 格式
        target_filepath = os.path.join(
            out_dir, 
            speaker.replace('-', '_') + '-' + utt_id.replace('-', '_') + '.wav'
        )
        
        # 如果已存在则跳过
        if os.path.exists(target_filepath):
            if i % 10 == 0:
                logging.info(f"文件已存在: {target_filepath}")
            continue
            
        shutil.copyfile(filepath, target_filepath)
    
    logging.info(f"复制完成，耗时: {time() - orig_time:.3f}s")


# 入口点
if __name__ == "__main__":
    # 示例用法
    plot_loss_acc(file='gru_checkpoints/losses_gru.txt', out_path='figs/loss_acc_gru.png')
    plot_metrics(
        log_file='checkpoints/losses.txt', 
        columns=['loss'], 
        out_path='figs/loss.png',
        labels=['Training Loss']
    )
    plot_speaker_metrics(file='checkpoints/acc_eer.txt', out_path='figs/acc_eer.png')