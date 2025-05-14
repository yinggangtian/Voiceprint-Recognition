# -*- coding:utf-8 -*-

# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s

import os
from glob import glob
from python_speech_features import fbank, delta
import librosa
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
import librosa
import constants as c
from constants import SAMPLE_RATE
from time import time
import sys
import soundfile as sf
from utils import find_files, vad  # 引入公共模块中的函数

np.set_printoptions(threshold=200)
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def read_audio(filename, sample_rate=SAMPLE_RATE):
    #audio, sr = librosa.load(os.path.normpath(filename), sr=sample_rate, mono=True)
    
    audio, sr = sf.read(filename)
    #print(audio)
    audio = vad(audio.flatten())
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = int(end_sec * SAMPLE_RATE)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    return audio

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]#均值/max（标准差，）

def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)   #filter_bank (num_frames , 64),energies (num_frames ,)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    #delta_1 = normalize_frames(delta_1)
    #delta_2 = normalize_frames(delta_2)

    #frames_features = np.hstack([filter_banks, delta_1, delta_2])    # (num_frames , 192)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)

def data_catalog(dataset_dir=c.DATASET_DIR, pattern='*.npy'):
    print(f"DEBUG: data_catalog 搜索路径: {dataset_dir}，模式: {pattern}")
    libri = pd.DataFrame()
    files = find_files(dataset_dir, pattern=pattern)
    print(f"DEBUG: 找到 {len(files)} 个文件")
    if len(files) > 0:
        print(f"DEBUG: 示例文件: {files[0]}")
    
    libri['filename'] = files
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    
    # 提取说话人ID
    if '**/*.wav' in pattern or '**/*.flac' in pattern:
        # 处理wav/flac文件，从路径结构中提取说话人ID
        # LibriSpeech格式: .../speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.wav
        libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-3])
    else:
        # 处理npy文件，从文件名中提取说话人ID
        # 期望npy文件格式: {speaker_id}_{recording_id}.npy
        libri['speaker_id'] = libri['filename'].apply(lambda x: os.path.basename(x).split('_')[0])
    
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    if num_speakers > 0:
        print(f"DEBUG: 发现的说话人: {libri['speaker_id'].unique()[:10]}...")
        print(f"DEBUG: 每个说话人的样本数量: ")
        speaker_counts = libri['speaker_id'].value_counts()
        # 修复：在Python 3中，.items()返回迭代器，不能直接切片
        for i, (speaker, count) in enumerate(speaker_counts.items()):
            if i >= 10:
                break
            print(f"    说话人 {speaker}: {count} 个样本")
    return libri


def preprocess_sync(wav_dir=c.WAV_DIR,out_dir=c.DATASET_DIR):
    os.makedirs(out_dir, exist_ok=True)
    libri = data_catalog(wav_dir, pattern='**/*.wav')  #'/Users/walle/PycharmProjects/Speech/coding/deep-speaker-master/audio/LibriSpeechSamples/train-clean-100/19'

    print("extract fbank from audio and save as npy, using multiprocessing pool........ ")
    
    for i in range(len(libri)):
        filename = libri[i:i+1]['filename'].values[0]
        speaker_id = libri[i:i+1]['speaker_id'].values[0]
        
        # 获取文件基本名称，移除路径和扩展名
        base_name = os.path.basename(filename).split('.')[0]
        
        # 创建NPY文件名: speaker_id_utterance_id.npy
        newfilename = f'{speaker_id}_{base_name}.npy'

        target_filename = os.path.join(out_dir,newfilename)
        target_filename = target_filename.replace('\\', '/')
        #print(f'-----{target_filename}')
    
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
        if feature.ndim != 3 or feature.shape[0] < c.NUM_FRAMES or feature.shape[1] !=64 or feature.shape[2] != 1:
            print('there is an error in file:',filename)
            continue

        print(f'save {target_filename} , {feature.shape}')

        np.save(target_filename, feature)

def prep(libri,out_dir=c.DATASET_DIR,name='0'):
    start_time = time()
    i=0
    for i in range(len(libri)):
        orig_time = time()
        filename = libri[i:i+1]['filename'].values[0]
        speaker_id = libri[i:i+1]['speaker_id'].values[0]
        
        # 对于LibriSpeech格式，创建格式为 "speaker_id_utterance_id.npy" 的文件名
        base_name = os.path.basename(filename).split('.')[0]  # 去掉扩展名
        newname = f"{speaker_id}_{base_name}.npy"
        target_filename = os.path.join(out_dir, newname)

        if os.path.exists(target_filename):
            if i % 10 == 0: print("task:{0} No.:{1} Exist File:{2}".format(name, i, filename))
            continue
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
        if feature.ndim != 3 or feature.shape[0] < c.NUM_FRAMES or feature.shape[1] !=64 or feature.shape[2] != 1:
            print('there is an error in file:',filename)
            continue
        np.save(target_filename, feature)
        if i % 100 == 0:
            print("task:{0} cost time per audio: {1:.3f}s No.:{2} File name:{3}".format(name, time() - orig_time, i, filename))
    print("task %s runs %d seconds. %d files" %(name, time()-start_time,i))


def preprocess_and_save(wav_dir=c.WAV_DIR,out_dir=c.DATASET_DIR):
    os.makedirs(out_dir, exist_ok=True)
    orig_time = time()
    libri = data_catalog(wav_dir, pattern='**/*.wav')  #'/Users/walle/PycharmProjects/Speech/coding/deep-speaker-master/audio/LibriSpeechSamples/train-clean-100/19'

    print("extract fbank from audio and save as npy, using multiprocessing pool........ ")
    p = Pool(5)
    patch = int(len(libri)/5)
    for i in range(5):
        if i < 4:
            slibri=libri[i*patch: (i+1)*patch]
        else:
            slibri = libri[i*patch:]
        print("task %s slibri length: %d" %(i, len(slibri)))
        p.apply_async(prep, args=(slibri,out_dir,i))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

    print("Extract audio features and save it as npy file, cost {0} seconds".format(time()-orig_time))
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def test():
    libri = data_catalog()
    filename = 'E:/mingde-AI/Deep_Speaker_exp3/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100/7312/92432/7312-92432-0000.wav'
    raw_audio = read_audio(filename)
    print(filename)
    feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
    print(filename)

if __name__ == '__main__':
    test()
    #preprocess_sync(wav_dir='E:/mingde-AI/Deep_Speaker_exp3/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100', out_dir='E:/mingde-AI/Deep_Speaker_exp3/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100-npy/')
    #preprocess_sync(wav_dir='E:/mingde-AI/Deep_Speaker_exp3/Deep_Speaker_exp/audio/LibriSpeechTest/test-clean', out_dir='E:/mingde-AI/Deep_Speaker_exp3/Deep_Speaker_exp/audio/LibriSpeechTest/test-clean-npy/')