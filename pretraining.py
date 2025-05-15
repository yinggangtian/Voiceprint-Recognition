#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from time import time

import constants as c
from utils import load_metadata, build_label_map, split_metadata, get_last_checkpoint, clean_old_checkpoints
from models import convolutional_model

# ─── 在任何 TensorFlow/Keras 导入之前，先屏蔽大部分日志 ───
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── 初始化模型函数 ───────────────────────────────────────────
def initialize_model(input_shape, no_of_speakers):
    """
    初始化模型。

    Args:
        input_shape: 输入数据的形状，例如 (c.NUM_MELS, None, c.CHANNELS)。
        no_of_speakers: 说话人数量，即分类数量。

    Returns:
        编译后的 Keras 模型。
    """
    base = convolutional_model(input_shape=input_shape) # 使用卷积模型作为基础
    x = Dense(no_of_speakers, activation='softmax', name='softmax_layer')(base.output) # 添加全连接层和 softmax 激活
    model = Model(inputs=base.input, outputs=x) # 定义模型输入输出
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']) # 使用 Adam 优化器和交叉熵损失
    logging.info(model.summary()) # 打印模型结构
    return model

# ─── 动态填充函数 ───────────────────────────────────────────
@tf.function
def pad_and_stack(melspecs, labels):
    """
    将一个批次内的 Mel 频谱图填充到相同的最大长度，并将它们堆叠起来。
    Args:
        melspecs: 批次 Mel 频谱图，形状为 (batch, num_mels, time)
        labels: 批次标签，one-hot，形状为 (batch, num_speakers)
    Returns:
        padded_melspecs: (batch, num_mels, max_time, 1)
        labels: (batch, num_speakers)
    """
    # 确保张量不是在Python循环中迭代
    # 直接使用tf.shape获取时间维度的最大值
    # 假设melspecs已经是批处理后的形状 [batch_size, mel_bands, time]
    if len(tf.shape(melspecs)) == 3:
        # 获取每个样本在第2维(时间维度)上的长度
        max_time = tf.shape(melspecs)[2]
        # 已经是批次张量，添加通道维度
        padded_melspecs = tf.expand_dims(melspecs, axis=-1)
    else:
        # 为了向后兼容，如果输入是批次列表（tf.RaggedTensor或类似结构）
        # 使用map_fn处理每个元素
        def get_time_dim(x):
            return tf.shape(x)[1]
        
        time_dims = tf.map_fn(get_time_dim, melspecs, fn_output_signature=tf.int32)
        max_time = tf.reduce_max(time_dims)
        
        def pad_spec(spec):
            pad_width = max_time - tf.shape(spec)[1]
            padded = tf.pad(spec, [[0, 0], [0, pad_width]])
            return padded
            
        padded_specs = tf.map_fn(pad_spec, melspecs, fn_output_signature=tf.float32)
        padded_melspecs = tf.expand_dims(padded_specs, axis=-1)
    
    return padded_melspecs, labels

# ─── 路径到加载器的修改版本 ───────────────────────────────────────────
def paths_to_loaders(metadata, melspec_dir, label_map, batch_size):
    """
    从元数据创建 TensorFlow Dataset。

    Args:
        metadata: 包含文件路径和说话人 ID 的 pandas DataFrame。
        melspec_dir: Mel 频谱图文件存储的目录。
        label_map: 将说话人 ID 映射到数字标签的字典。
        batch_size: 所需的批次大小。

    Returns:
        一个 TensorFlow Dataset，它产生 (padded_melspecs, labels) 批次。
    """
    def generator():
        """
        生成器函数，用于按需加载 Mel 频谱图和标签。
        """
        for index, row in metadata.iterrows(): # 遍历元数据中的每一行
            try:
                # 直接使用说话人ID和文件名构建melspec文件路径
                # 这适用于prepare_test_data.py生成的测试数据
                speaker_id = str(row['speaker_id'])
                
                if 'file_path' in row:
                    # 从file_path中提取文件名部分
                    file_path = row['file_path']
                    file_name = os.path.basename(file_path)
                    file_name_no_ext = os.path.splitext(file_name)[0]
                    # 尝试直接在melspec_dir目录下查找文件
                    melspec_path = os.path.join(melspec_dir, file_name_no_ext + '.npy')
                    
                    # 检查文件是否存在，如果不存在，尝试只使用说话人ID部分
                    if not os.path.exists(melspec_path) and 'subset' in row:
                        # 按数据下载工具生成的结构尝试
                        subset = row['subset']
                        melspec_path = os.path.join(melspec_dir, subset, f"{speaker_id}_{file_name_no_ext}.npy")
                else:
                    # 向后兼容，如果直接有filename列
                    melspec_path = os.path.join(melspec_dir, row.get('filename', '') + '.npy')
                
                # 加载梅尔频谱图
                melspec = np.load(melspec_path)
                # 获取说话人对应的数字标签
                label = label_map[row['speaker_id']]
                # 将数字标签转换为 one-hot 编码
                one_hot_label = tf.one_hot(label, depth=len(label_map))
                # 产生 Mel 频谱图和 one-hot 编码的标签
                yield melspec, one_hot_label
            except (FileNotFoundError, KeyError) as e:
                logging.warning(f"跳过文件 {melspec_path if 'melspec_path' in locals() else '未知路径'}: {e}")

    # Use tf.data.Dataset.from_generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(c.NUM_MELS, None), dtype=tf.float32),  # Dynamic time dimension # 指定 Mel 频谱图的形状，时间维度为 None，表示动态长度
            tf.TensorSpec(shape=(len(label_map),), dtype=tf.float32), #one hot # 指定标签的形状
        )
    )
    # 使用padded_batch处理不同长度的梅尔频谱图
    dataset = dataset.padded_batch(
        batch_size, 
        padded_shapes=((c.NUM_MELS, None), (len(label_map),)),  # 指定填充形状
        padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))  # 填充值
    ).map(pad_and_stack) # 使用 pad_and_stack 函数添加通道维度并处理张量

    return dataset # 返回构建好的 TensorFlow Dataset

# ─── 训练模型函数 ───────────────────────────────────────────
def train_model(model, train_loader, test_loader, max_steps=None):
    """
    训练模型。

    Args:
        model: 要训练的 Keras 模型。
        train_loader: 训练数据加载器（TensorFlow Dataset）。
        test_loader: 测试数据加载器（TensorFlow Dataset）。
        max_steps: 最大训练步数。如果为 None，则无限训练。
    """
    steps = 0 # 初始化训练步数
    last_ckpt = get_last_checkpoint(c.PRE_CHECKPOINT_FOLDER) # 获取最后一个检查点
    if last_ckpt:
        model.load_weights(last_ckpt) # 如果存在检查点，则加载权重
        steps = int(last_ckpt.split('_')[-2]) # 从检查点文件名中提取步数

    start_time = time()

    while True: # 无限循环，直到达到所需的训练轮数或满足其他停止条件
        for x, y in train_loader: # Iterate through the tf.data.Dataset # 从训练数据加载器中获取一个批次的训练数据
            loss, acc = model.train_on_batch(x, y) # 在该批次上进行训练，并获取损失和准确率
            
            # 显示进度信息（如果设置了最大步数）
            if max_steps is not None:
                progress_percent = (steps / max_steps) * 100
                elapsed_time = time() - start_time
                estimated_total_time = elapsed_time / (steps + 1) * max_steps
                estimated_remaining_time = estimated_total_time - elapsed_time
                logging.info(f"Step {steps}/{max_steps} ({progress_percent:.1f}%) — train loss={loss:.4f}, acc={acc:.4f} — Est. remaining: {estimated_remaining_time:.2f}s") # 打印训练信息
            else:
                logging.info(f"Step {steps} — train loss={loss:.4f}, acc={acc:.4f}") # 打印训练信息

            if steps % c.TEST_PER_EPOCHS == 0: # 每 c.TEST_PER_EPOCHS 步进行一次测试
                for xt, yt in test_loader:
                    tl, ta = model.test_on_batch(xt, yt) # 在测试集上评估模型
                    logging.info(f"Step {steps} — test loss={tl:.4f}, acc={ta:.4f}") # 打印测试信息
                    break # Only test on the first batch for efficiency # 为了效率，仅在第一个测试批次上进行测试

            if steps % c.SAVE_PER_EPOCHS == 0: # 每 c.SAVE_PER_EPOCHS 步保存一次模型
                clean_old_checkpoints(c.PRE_CHECKPOINT_FOLDER, keep_latest=3) # 清理旧的检查点，仅保留最新的 3 个
                model.save_weights(
                    os.path.join(c.PRE_CHECKPOINT_FOLDER, f"model_{steps}_{loss:.4f}.weights.h5") # 保存模型权重
                )
            
            steps += 1 # 增加训练步数
            
            # 检查是否达到最大步数
            if max_steps is not None and steps >= max_steps:
                logging.info(f"已达到最大步数 ({max_steps})。训练完成。")
                return

# ─── 主入口 ───────────────────────────────────────────
def main():
    """
    主函数，用于执行整个训练流程。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='预训练声纹识别模型。')
    parser.add_argument('--max_steps', type=int, default=None, help='最大训练步数')
    args = parser.parse_args()
    
    logging.basicConfig(
        handlers=[logging.StreamHandler(sys.stdout)], # 将日志输出到标准输出
        level=logging.INFO, # 设置日志级别为 INFO
        format="%(asctime)s [%(levelname)s] %(message)s" # 设置日志格式
    )

    meta = load_metadata("metadata_small.csv") # 加载元数据
    train_meta, test_meta = split_metadata(meta, train_frac=0.8) # 将元数据分割为训练集和测试集

    label_map = build_label_map(meta) # 构建标签映射，将说话人 ID 映射到数字标签
    train_loader = paths_to_loaders(train_meta, "melspec_small", label_map, c.BATCH_SIZE) # 创建训练数据加载器
    test_loader  = paths_to_loaders(test_meta,  "melspec_small", label_map, c.BATCH_SIZE) # 创建测试数据加载器

    # 新的输入形状
    input_shape = (c.NUM_MELS, None, c.CHANNELS)  # e.g. (64, None, 1) # 定义模型的输入形状，时间维度为 None，表示动态长度
    model = initialize_model(input_shape, no_of_speakers=len(label_map)) # 初始化模型
    train_model(model, train_loader, test_loader, max_steps=args.max_steps) # 训练模型

if __name__ == "__main__":
    main() # 执行主函数
