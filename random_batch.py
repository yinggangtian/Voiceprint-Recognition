"""
   filename                             chapter_id speaker_id dataset_id
0  1272/128104/1272-128104-0000.wav     128104       1272  dev-clean
1  1272/128104/1272-128104-0001.wav     128104       1272  dev-clean
2  1272/128104/1272-128104-0002.wav     128104       1272  dev-clean
3  1272/128104/1272-128104-0003.wav     128104       1272  dev-clean
4  1272/128104/1272-128104-0004.wav     128104       1272  dev-clean
5  1272/128104/1272-128104-0005.wav     128104       1272  dev-clean
6  1272/128104/1272-128104-0006.wav     128104       1272  dev-clean
7  1272/128104/1272-128104-0007.wav     128104       1272  dev-clean
8  1272/128104/1272-128104-0008.wav     128104       1272  dev-clean
9  1272/128104/1272-128104-0009.wav     128104       1272  dev-clean
"""

import numpy as np
import pandas as pd

import constants as c
from pre_process import data_catalog


def clipped_audio(x, num_frames=c.NUM_FRAMES):
    # 处理 num_frames 为 None 的情况
    if num_frames is None:
        # 如果未指定帧数，直接返回原始数据
        return x
    
    if x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x


class MiniBatch:
    def __init__(self, libri, batch_size, unique_speakers=None):    #libri['filename']，libri['chapter_id']，libri['speaker_id']，libri['dataset_id']
        # indices = np.random.choice(len(libri), size=batch_size, replace=False)
        # [anc1, anc2, anc3, pos1, pos2, pos3, neg1, neg2, neg3]
        # [sp1, sp2, sp3, sp1, sp2, sp3, sp4, sp5, sp6]
        if unique_speakers is None:
            unique_speakers = list(libri['speaker_id'].unique())
        num_triplets = batch_size

        anchor_batch = None
        positive_batch = None
        negative_batch = None
        for ii in range(num_triplets):
            two_different_speakers = np.random.choice(unique_speakers, size=2, replace=False)
            anchor_positive_speaker = two_different_speakers[0]
            negative_speaker = two_different_speakers[1]
            anchor_positive_file = libri[libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
            anchor_df = pd.DataFrame(anchor_positive_file[0:1])
            anchor_df['training_type'] = 'anchor'
            positive_df = pd.DataFrame(anchor_positive_file[1:2])
            positive_df['training_type'] = 'positive'
            negative_df = libri[libri['speaker_id'] == negative_speaker].sample(n=1)
            negative_df['training_type'] = 'negative'

            if anchor_batch is None:
                anchor_batch = anchor_df.copy()
            else:
                anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)
            if positive_batch is None:
                positive_batch = positive_df.copy()
            else:
                positive_batch = pd.concat([positive_batch, positive_df], axis=0)
            if negative_batch is None:
                negative_batch = negative_df.copy()
            else:
                negative_batch = pd.concat([negative_batch, negative_df], axis=0)

        self.libri_batch = pd.DataFrame(pd.concat([anchor_batch, positive_batch, negative_batch], axis=0))
        self.num_triplets = num_triplets

    def to_inputs(self):
        new_x = []
        max_frames = 0
        feature_dim = 64  # 标准特征维度
        channel_dim = 1   # 标准通道数
        
        # 第一轮：加载所有音频并找出最大帧数
        loaded_audios = []
        for i in range(len(self.libri_batch)):
            filename = self.libri_batch[i:i + 1]['filename'].values[0]
            x = np.load(filename)
            x_clipped = clipped_audio(x)
            
            # 确保数据形状一致
            if len(x_clipped.shape) == 1:
                # 如果只有一个维度，reshape 为 (frames, features)
                frames = len(x_clipped)
                x_clipped = x_clipped.reshape(frames, 1)
            
            if len(x_clipped.shape) == 2:
                # 如果只有两个维度 (frames, features)，添加通道维度
                x_clipped = np.expand_dims(x_clipped, axis=2)
                
            # 更新特征和通道维度 (如果需要)
            feature_dim = max(feature_dim, x_clipped.shape[1]) 
            channel_dim = max(channel_dim, x_clipped.shape[2])
            
            loaded_audios.append(x_clipped)
            max_frames = max(max_frames, x_clipped.shape[0])
            
        # 第二轮：应用填充（padding）使所有样本具有相同长度
        for x_clipped in loaded_audios:
            # 创建填充后的数组 - 确保所有样本具有相同的形状
            padded = np.zeros((max_frames, feature_dim, channel_dim), dtype=np.float32)
            
            # 复制原始数据，安全地处理各种维度
            frames_to_copy = min(x_clipped.shape[0], max_frames)
            
            # 根据输入数据的维度安全地处理
            if len(x_clipped.shape) == 3:
                feat_to_copy = min(x_clipped.shape[1], feature_dim)
                chan_to_copy = min(x_clipped.shape[2], channel_dim)
                padded[:frames_to_copy, :feat_to_copy, :chan_to_copy] = x_clipped[:frames_to_copy, :feat_to_copy, :chan_to_copy]
            elif len(x_clipped.shape) == 2:
                feat_to_copy = min(x_clipped.shape[1], feature_dim)
                padded[:frames_to_copy, :feat_to_copy, 0] = x_clipped[:frames_to_copy, :feat_to_copy]
            else:
                # 处理极端情况
                padded[:frames_to_copy, 0, 0] = x_clipped[:frames_to_copy]
                
            new_x.append(padded)
        
        # 将列表转换为numpy数组
        try:
            x = np.array(new_x) #(batchsize, max_frames, feature_dim, channel_dim)
            print(f"批次数据形状: {x.shape}")
        except ValueError as e:
            print("创建数组失败，检查形状:", e)
            # 打印每个数组的形状以便调试
            for i, arr in enumerate(new_x):
                print(f"数组 {i} 形状: {arr.shape}")
            raise
        y = self.libri_batch['speaker_id'].values

        # anchor examples [speakers] == positive examples [speakers]
        np.testing.assert_array_equal(y[0:self.num_triplets], y[self.num_triplets:2 * self.num_triplets])

        return x, y


def stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE,unique_speakers=None):
    mini_batch = MiniBatch(libri, batch_size,unique_speakers)
    return mini_batch


def main():
    libri = data_catalog(c.WAV_DIR)
    batch = stochastic_mini_batch(libri, c.BATCH_SIZE)

    x, y = batch.to_inputs()
    print(x.shape,y.shape)


if __name__ == '__main__':
    main()
