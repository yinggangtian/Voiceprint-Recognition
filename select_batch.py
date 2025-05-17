
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# batchisize 32*3 : train on triplet: 3.3s/steps , softmax pre train: 3.1 s/steps  ,select_best_batch
# local: load pkl time 0.00169s - > open file time 4.2e-05s pickle loading time 0.00227s
# server: load pkl time 0.0389s -> open file  time 6.1e-05s pickle load time 0.0253s



import pandas as pd
import random
import numpy as np
import constants as c
from utils import get_last_checkpoint
from models import convolutional_model
from triplet_loss import deep_speaker_loss
from pre_process import data_catalog
import heapq
import threading
from time import time, sleep

alpha = c.ALPHA

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)
    return s

def matrix_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.dot(x1, x2.T)
    return mul

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    """
    处理音频数据，支持多种模式的可变长度输入处理
    
    Args:
        x: 输入的音频特征数据，形状为 (frames, features, channels)
        num_frames: 目标帧数，None 表示使用原始长度
        
    Returns:
        处理后的音频特征数据
    """
    # 打印调试信息
    original_shape = x.shape
    
    # 确保x至少是2D数组
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    
    # 处理 num_frames 为 None 的情况
    if num_frames is None:
        # 如果未指定帧数，直接返回原始数据（确保3D）
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x
    
    # 获取PAD_MODE
    pad_mode = getattr(c, 'PAD_MODE', 'none')
    
    # 根据目标帧数处理音频
    if x.shape[0] > num_frames:
        # 数据超过目标帧数，需要裁剪
        margin = min(20, x.shape[0] - num_frames)  # 防止margin超出范围
        if margin > 0:
            # 有边缘空间，从中间随机选择
            bias = np.random.randint(0, margin)
            clipped_x = x[bias: num_frames + bias]
        else:
            # 没有边缘空间，直接从开头裁剪
            clipped_x = x[:num_frames]
    else:
        # 数据长度不足目标帧数，需要填充
        clipped_x = x  # 默认保持原始数据
        
        if pad_mode == 'zero':
            # 零填充模式
            pad_shape = list(x.shape)
            pad_shape[0] = num_frames - x.shape[0]
            padding = np.zeros(pad_shape, dtype=x.dtype)
            clipped_x = np.concatenate([x, padding], axis=0)
            
        elif pad_mode == 'repeat':
            # 重复最后一帧模式
            if x.shape[0] > 0:  # 确保有帧可以重复
                repeats_needed = int(np.ceil((num_frames - x.shape[0]) / x.shape[0]))
                if repeats_needed > 0:
                    repeat_frames = np.repeat(x, repeats_needed + 1, axis=0)
                    clipped_x = repeat_frames[:num_frames]
            
        elif pad_mode == 'mirror':
            # 镜像填充模式
            if x.shape[0] > 1:  # 确保有足够的帧进行镜像
                # 创建镜像序列
                mirror = np.flip(x, axis=0)
                combined = np.concatenate([x, mirror], axis=0)
                # 重复直到足够长
                repeats_needed = int(np.ceil(num_frames / combined.shape[0]))
                if repeats_needed > 0:
                    repeated = np.concatenate([combined] * (repeats_needed + 1), axis=0)
                    clipped_x = repeated[:num_frames]
    
    # 确保输出是3D张量
    if len(clipped_x.shape) == 2:
        clipped_x = np.expand_dims(clipped_x, axis=2)
    
    # 打印最终形状
    if clipped_x.shape != original_shape:
        print(f"形状变化: {original_shape} -> {clipped_x.shape}")
        
    return clipped_x

spk_utt_index = {}
def preprocess(unique_speakers, spk_utt_dict,candidates=c.CANDIDATES_PER_BATCH):
    files = []
    flag = False if len(unique_speakers) > candidates/2 else True
    speakers = np.random.choice(unique_speakers, size=int(candidates/2), replace=flag)
    for speaker in speakers:
        index=0
        ll = len(spk_utt_dict[speaker])
        if speaker in spk_utt_index:
            index = spk_utt_index[speaker] % ll
        files.append(spk_utt_dict[speaker][index])
        files.append(spk_utt_dict[speaker][(index+1)%ll])
        spk_utt_index[speaker] = (index + 2) % ll
        '''
    for ii in range(int(candidates/2)):
        utts = libri[libri['speaker_id'] == speakers[ii]].sample(n=2, replace=False)
        files = files.append(utts)
        #print("sampling utterance time {0:.5}s".format(time() - orig_time))
        #orig_time = time()
    '''
    x = []
    labels = []
    
    # 第一轮：加载所有音频并找出最大帧数、特征维度和通道数
    loaded_audios = []
    max_frames = 0
    feature_dim = 64  # 标准特征维度
    channel_dim = 1   # 标准通道数
    
    for file in files:
        x_ = np.load(file)
        x_ = clipped_audio(x_)
        
        # 确保数据有三个维度
        if len(x_.shape) == 1:
            # 如果只有一个维度，reshape 为 (frames, features)
            frames = len(x_)
            x_ = x_.reshape(frames, 1)
            
        if len(x_.shape) == 2:
            # 如果只有两个维度 (frames, features)，添加通道维度
            x_ = np.expand_dims(x_, axis=2)
            
        # 更新特征和通道维度 (如果需要)
        feature_dim = max(feature_dim, x_.shape[1]) if len(x_.shape) > 1 else feature_dim
        channel_dim = max(channel_dim, x_.shape[2]) if len(x_.shape) > 2 else channel_dim
        
        loaded_audios.append(x_)
        max_frames = max(max_frames, x_.shape[0])
        labels.append(file.split("/")[-1].split("_")[0])
    
    # 第二轮：应用填充（padding）使所有样本具有相同长度和形状
    for x_ in loaded_audios:
        # 创建填充后的数组 - 确保所有样本具有相同的形状
        padded = np.zeros((max_frames, feature_dim, channel_dim), dtype=np.float32)
        
        # 复制原始数据
        frames_to_copy = min(x_.shape[0], max_frames)
        
        # 根据输入数据的维度安全地处理
        if len(x_.shape) == 3:
            feat_to_copy = min(x_.shape[1], feature_dim)
            chan_to_copy = min(x_.shape[2], channel_dim)
            padded[:frames_to_copy, :feat_to_copy, :chan_to_copy] = x_[:frames_to_copy, :feat_to_copy, :chan_to_copy]
        elif len(x_.shape) == 2:
            feat_to_copy = min(x_.shape[1], feature_dim)
            padded[:frames_to_copy, :feat_to_copy, 0] = x_[:frames_to_copy, :feat_to_copy]
        else:  # 一维数组情况
            padded[:frames_to_copy, 0, 0] = x_[:frames_to_copy]
            
        x.append(padded)
        
    # 添加调试信息，检查最终批次的形状
    if x:
        print(f"Batch shape after padding: {np.array(x).shape}")

    return np.array(x), np.array(labels)

stack = []
def create_data_producer(unique_speakers, spk_utt_dict,candidates=c.CANDIDATES_PER_BATCH):
    producer = threading.Thread(target=addstack, args=(unique_speakers, spk_utt_dict,candidates))
    producer.setDaemon(True)
    producer.start()

def addstack(unique_speakers, spk_utt_dict,candidates=c.CANDIDATES_PER_BATCH):
    data_produce_step = 0
    while True:
        if len(stack) >= c.DATA_STACK_SIZE:
            sleep(0.01)
            continue

        orig_time = time()
        feature, labels = preprocess(unique_speakers, spk_utt_dict, candidates)
        #print("pre-process one batch data costs {0:.4f} s".format(time() - orig_time))
        stack.append((feature, labels))

        data_produce_step += 1
        if data_produce_step % 100 == 0:
            for spk in unique_speakers:
                np.random.shuffle(spk_utt_dict[spk])

def getbatch():
    while True:
        if len(stack) == 0:
            continue
        return stack.pop(0)

hist_embeds = None
hist_labels = None
hist_features = None
hist_index = 0
hist_table_size = c.HIST_TABLE_SIZE
def best_batch(model, batch_size=c.BATCH_SIZE, candidates=c.CANDIDATES_PER_BATCH):
    """
    从候选数据中选择最佳的批次进行训练。
    支持可变长度输入和固定长度输入。
    
    Args:
        model: 训练模型 (CNN)
        batch_size: 批次大小
        candidates: 候选项数量
        
    Returns:
        features: 选择的特征批次数据
        labels: 对应的标签
    """
    orig_time = time()
    global hist_embeds, hist_features, hist_labels, hist_index, hist_table_size
    features, labels = getbatch()
    print("获取批次耗时 {0:.3}s".format(time() - orig_time))
    
    # 打印原始批次形状
    print(f"原始批次数量: {len(features)}, 形状: {features[0].shape if len(features) > 0 else 'N/A'}")
    
    # 检测是否所有输入具有相同形状
    shapes = [f.shape for f in features]
    is_same_shape = all(s == shapes[0] for s in shapes)
    
    # 如果不是相同形状且需要处理，进行形状调整
    if not is_same_shape:
        # 找出最大帧数和特征维度
        max_frames = max(s[0] for s in shapes)
        feature_dim = max(s[1] for s in shapes) if all(len(s) > 1 for s in shapes) else shapes[0][1]
        channel_dim = max(s[2] for s in shapes) if all(len(s) > 2 for s in shapes) else 1
        
        # 打印维度信息
        print(f"统一形状到: 帧数={max_frames}, 特征维度={feature_dim}, 通道数={channel_dim}")
        
        padded_features = []
        for i, x in enumerate(features):
            if len(x.shape) < 3:
                # 如果维度不够，添加缺失的维度
                if len(x.shape) == 1:
                    x = x.reshape(x.shape[0], 1, 1)
                elif len(x.shape) == 2:
                    x = np.expand_dims(x, axis=2)
            
            # 创建填充后的数组
            padded_x = np.zeros((max_frames, feature_dim, channel_dim), dtype=np.float32)
            
            # 复制原始数据
            frames_to_copy = min(x.shape[0], max_frames)
            feat_to_copy = min(x.shape[1], feature_dim)
            chan_to_copy = min(x.shape[2], channel_dim)
            
            padded_x[:frames_to_copy, :feat_to_copy, :chan_to_copy] = x[:frames_to_copy, :feat_to_copy, :chan_to_copy]
            padded_features.append(padded_x)
        
        # 更新特征
        features = np.array(padded_features)
    else:
        # 确保features是numpy数组
        features = np.array(features)
    
    # 打印最终形状
    print(f"处理后批次形状: {features.shape}")
    
    orig_time = time()
    # 确保模型输入有效且形状一致
    try:
        embeds = model.predict_on_batch(features)
        print(f"嵌入向量形状: {embeds.shape}")
    except Exception as e:
        print(f"前向传播错误: {e}")
        # 如果预测失败，尝试一个样本测试特征形状兼容性
        test_shape = model.predict_on_batch(np.expand_dims(features[0], 0)).shape
        print(f"模型需要的输入形状: {features[0].shape}, 输出形状: {test_shape}")
        raise
    
    print("前向传播耗时 {0:.3}s".format(time()-orig_time))

    # 更新历史表
    if hist_embeds is None:
        hist_features = np.copy(features)
        hist_labels = np.copy(labels)
        hist_embeds = np.copy(embeds)
    else:
        if len(hist_labels) < hist_table_size*candidates:
            hist_features = np.concatenate((hist_features, features), axis=0)
            hist_labels = np.concatenate((hist_labels, labels), axis=0)
            hist_embeds = np.concatenate((hist_embeds, embeds), axis=0)
        else:
            # 更新历史表
            start_idx = hist_index * candidates
            end_idx = min((hist_index + 1) * candidates, len(hist_features))
            
            # 确保索引在有效范围内
            if end_idx > start_idx:
                # 检查维度匹配
                if features.shape[1:] != hist_features[0].shape[1:]:
                    print(f"警告: 特征形状不匹配, 历史={hist_features[0].shape}, 当前={features.shape}")
                    # 调整历史特征以匹配新特征
                    adjusted_features = []
                    for i, x in enumerate(features):
                        if i >= (end_idx - start_idx):
                            break
                        
                        # 创建调整后的特征
                        adjusted_x = np.zeros_like(hist_features[start_idx + i])
                        min_frames = min(x.shape[0], hist_features[start_idx + i].shape[0])
                        min_feats = min(x.shape[1], hist_features[start_idx + i].shape[1])
                        min_chans = min(x.shape[2], hist_features[start_idx + i].shape[2])
                        
                        adjusted_x[:min_frames, :min_feats, :min_chans] = x[:min_frames, :min_feats, :min_chans]
                        adjusted_features.append(adjusted_x)
                    
                    if adjusted_features:
                        features_to_update = np.array(adjusted_features)
                        hist_features[start_idx:start_idx + len(features_to_update)] = features_to_update
                else:
                    # 形状匹配时直接更新
                    copy_size = min(len(features), end_idx - start_idx)
                    hist_features[start_idx:start_idx + copy_size] = features[:copy_size]
                
                # 更新标签和嵌入
                copy_size = min(len(labels), end_idx - start_idx)
                hist_labels[start_idx:start_idx + copy_size] = labels[:copy_size]
                
                copy_size = min(len(embeds), end_idx - start_idx)
                hist_embeds[start_idx:start_idx + copy_size] = embeds[:copy_size]

    # 更新索引
    hist_index = (hist_index+1) % hist_table_size

    # 初始化批次
    anchor_batch = []
    positive_batch = []
    negative_batch = []
    anchor_labs, positive_labs, negative_labs = [], [], []

    orig_time = time()
    # 随机选择说话人
    unique_labels = np.unique(hist_labels)
    if len(unique_labels) < batch_size/2:
        print(f"警告: 可用说话人({len(unique_labels)})少于所需数量({batch_size/2}), 将使用重复说话人")
    
    # 确保有足够的说话人，必要时使用replace=True
    replace_speakers = len(unique_labels) < batch_size/2
    anh_speakers = np.random.choice(unique_labels, int(batch_size/2), replace=replace_speakers)
    
    # 构建说话人索引字典
    anchs_index_dict = {}
    inds_set = []
    for spk in anh_speakers:
        anhinds = np.argwhere(hist_labels==spk).flatten()
        if len(anhinds) > 0:
            anchs_index_dict[spk] = anhinds
            inds_set.extend(anhinds)
    inds_set = list(set(inds_set))

    # 计算嵌入相似度
    if len(inds_set) > 0:
        speakers_embeds = hist_embeds[inds_set]
        sims = matrix_cosine_similarity(speakers_embeds, hist_embeds)
        
        print('Beginning to select triplets...')
        valid_triplets = 0
        
        for ii in range(int(batch_size/2)):
            if ii >= len(anh_speakers):
                print(f"警告: 索引 {ii} 超出说话人列表范围 {len(anh_speakers)}")
                break
                
            speaker = anh_speakers[ii]
            if speaker not in anchs_index_dict:
                print(f"警告: 说话人 {speaker} 没有足够样本")
                continue
                
            inds = anchs_index_dict[speaker]
            if len(inds) < 1:
                print(f"警告: 说话人 {speaker} 没有样本")
                continue
                
            # 打乱索引
            np.random.shuffle(inds)
            anchor_index = inds[0]
            
            # 查找正样本索引
            pinds = []
            for jj in range(1, len(inds)):
                if jj >= len(inds):
                    break
                if np.array_equal(hist_features[anchor_index], hist_features[inds[jj]]):
                    continue
                pinds.append(inds[jj])

            # 如果没有足够的正样本，跳过这个三元组
            if len(pinds) < 1:
                print(f"警告: 说话人 {speaker} 没有足够的正样本")
                continue

            # 计算最不相似的正样本
            sap = sims[ii][pinds]
            min_saps = heapq.nsmallest(min(2, len(sap)), sap)
            
            # 找到对应索引
            pos0_index = pinds[np.argwhere(sap == min_saps[0]).flatten()[0]]
            if len(pinds) > 1 and len(min_saps) > 1:
                pos1_index = pinds[np.argwhere(sap == min_saps[1]).flatten()[0]]
            else:
                pos1_index = pos0_index

            # 查找负样本索引
            ninds = np.argwhere(hist_labels != speaker).flatten()
            if len(ninds) < 1:
                print(f"警告: 没有足够的负样本")
                continue
                
            # 计算最相似的负样本
            san = sims[ii][ninds]
            max_sans = heapq.nlargest(min(2, len(san)), san)
            
            # 找到对应索引
            neg0_index = ninds[np.argwhere(san == max_sans[0]).flatten()[0]]
            if len(ninds) > 1 and len(max_sans) > 1:
                neg1_index = ninds[np.argwhere(san == max_sans[1]).flatten()[0]]
            else:
                neg1_index = neg0_index

            # 添加到批次
            anchor_batch.append(hist_features[anchor_index])
            anchor_batch.append(hist_features[anchor_index])
            positive_batch.append(hist_features[pos0_index])
            positive_batch.append(hist_features[pos1_index])
            negative_batch.append(hist_features[neg0_index])
            negative_batch.append(hist_features[neg1_index])

            # 添加标签
            anchor_labs.append(hist_labels[anchor_index])
            anchor_labs.append(hist_labels[anchor_index])
            positive_labs.append(hist_labels[pos0_index])
            positive_labs.append(hist_labels[pos1_index])
            negative_labs.append(hist_labels[neg0_index])
            negative_labs.append(hist_labels[neg1_index])
            
            valid_triplets += 2  # 每次循环添加两个三元组
    else:
        print("警告: 没有有效的说话人索引")

    # 确保所有批次都有元素
    if not anchor_batch or not positive_batch or not negative_batch:
        print("警告: 批次为空，使用随机特征")
        # 使用随机特征作为后备
        random_indices = np.random.choice(len(features), min(batch_size*3, len(features)), replace=len(features) < batch_size*3)
        random_batch = features[random_indices]
        random_labs = labels[random_indices]
        
        # 分割成三等份
        third = len(random_batch) // 3
        anchor_batch = random_batch[:third]
        positive_batch = random_batch[third:2*third]
        negative_batch = random_batch[2*third:3*third]
        
        anchor_labs = random_labs[:third]
        positive_labs = random_labs[third:2*third]
        negative_labs = random_labs[2*third:3*third]

    # 确保三个批次有相同的大小
    min_size = min(len(anchor_batch), len(positive_batch), len(negative_batch))
    anchor_batch = anchor_batch[:min_size]
    positive_batch = positive_batch[:min_size]
    negative_batch = negative_batch[:min_size]
    
    # 同样确保标签也有相同的大小
    anchor_labs = anchor_labs[:min_size]
    positive_labs = positive_labs[:min_size]
    negative_labs = negative_labs[:min_size]
    
    # 打印批次大小信息
    print(f"最终三元组数量: {min_size}")
    print(f"Anchor batch: {len(anchor_batch)}, Positive batch: {len(positive_batch)}, Negative batch: {len(negative_batch)}")
    
    # 组合批次
    if anchor_batch and positive_batch and negative_batch:
        # 确保三个批次中的元素数量完全相同，且总数是3的倍数
        min_size = min(len(anchor_batch), len(positive_batch), len(negative_batch))
        # 确保是偶数长度，以保证一致的批次大小
        if min_size % 2 != 0:
            min_size -= 1
            
        anchor_batch = anchor_batch[:min_size]
        positive_batch = positive_batch[:min_size]
        negative_batch = negative_batch[:min_size]
        
        # 同样确保标签也保持一致
        anchor_labs = anchor_labs[:min_size]
        positive_labs = positive_labs[:min_size]
        negative_labs = negative_labs[:min_size]
        
        # 先转换为numpy数组再连接，保证形状一致
        anchor_np = np.array(anchor_batch)
        positive_np = np.array(positive_batch)
        negative_np = np.array(negative_batch)
        
        # 严格按照 [anchor, positive, negative] 的顺序连接
        batch = np.concatenate([anchor_np, positive_np, negative_np], axis=0)
        # 合并标签并打印信息
        labs = np.array(anchor_labs)
        
        # 打印最终批次信息
        print(f"最终三元组数量: {min_size}")
        print(f"Anchor: {anchor_np.shape}, Positive: {positive_np.shape}, Negative: {negative_np.shape}")
        print(f"合并后批次形状: {batch.shape}, 标签形状: {labs.shape}")
        
        # 确保批次大小是3的倍数
        if len(batch) % 3 != 0:
            print(f"警告: 调整批次大小以确保是3的倍数")
            truncate_to = (len(batch) // 3) * 3
            batch = batch[:truncate_to]
            labs = labs[:truncate_to//3]  # 标签只对应anchor部分
            print(f"调整后批次形状: {batch.shape}, 标签形状: {labs.shape}")
        
        # XLA编译兼容性检查
        third = len(batch) // 3
        if len(batch) != third * 3:
            print(f"错误: 批次大小 {len(batch)} 不是3的精确倍数，这在XLA编译模式下会导致问题")
            # 再次修正
            batch = batch[:third * 3]
            
        # 打印最终的批次形状信息（调试用）
        print(f"最终批次形状: {batch.shape}, 每部分大小: {third}")
        print(f"批次大小检查: anchor={batch[:third].shape}, positive={batch[third:2*third].shape}, negative={batch[2*third:].shape}")
        
        print("选择最佳批次耗时 {0:.3}s".format(time() - orig_time))
        return batch, labs
    else:
        print("错误: 无法创建有效批次，使用随机批次作为后备方案")
        # 确保随机批次大小是3的精确倍数
        required_samples = ((batch_size * 3) // 3) * 3  # 确保是3的倍数
        
        # 检查是否有足够的样本
        if len(features) < required_samples:
            print(f"警告: 样本不足，需要使用重复采样 (需要:{required_samples}, 可用:{len(features)})")
            random_indices = np.random.choice(len(features), required_samples, replace=True)
        else:
            random_indices = np.random.choice(len(features), required_samples, replace=False)
        
        random_batch = features[random_indices]
        random_labs = labels[random_indices[:required_samples//3]]  # 只取前1/3作为标签
        
        # 确认大小
        print(f"随机批次形状: {random_batch.shape}, 标签形状: {random_labs.shape}")
        print(f"批次大小检查: 总数={len(random_batch)}, 每部分={len(random_batch)//3}")
        
        return random_batch, random_labs

if __name__ == '__main__':
    model = convolutional_model()
    model.compile(optimizer='adam', loss=deep_speaker_loss)
    last_checkpoint = get_last_checkpoint(c.CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        print('[DONE]')
    libri = data_catalog(c.DATASET_DIR)
    unique_speakers = libri['speaker_id'].unique()
    labels = libri['speaker_id'].values
    files = libri['filename'].values
    spk_utt_dict = {}
    for i in range(len(unique_speakers)):
        spk_utt_dict[unique_speakers[i]] = []

    for i in range(len(labels)):
        spk_utt_dict[labels[i]].append(files[i])

    create_data_producer(unique_speakers,spk_utt_dict)
    for i in range(100):
        x, y = best_batch(model)
        print(x.shape)
        #print(y)