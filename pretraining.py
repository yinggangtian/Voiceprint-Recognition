#!/usr/bin/env python3
import os
import sys
import logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
def pad_and_stack(batch_data):
    """
    将一个批次内的 Mel 频谱图填充到相同的最大长度，并将它们堆叠起来。

    Args:
        batch_data: 一个批次的数据，它是一个元组列表，其中每个元组是 (melspec, label)。
                    melspec 的形状为 (c.NUM_MELS, time_frames)。

    Returns:
        一个元组 (padded_melspecs, labels)，其中：
            - padded_melspecs 的形状为 (batch_size, c.NUM_MELS, max_time_frames, c.CHANNELS)。
            - labels 的形状为 (batch_size, num_speakers)。
    """
    melspecs, labels = batch_data # 从批次数据中解压出 Mel 频谱图和标签
    # Find the maximum time_frames in the batch
    max_time_frames = max(spec.shape[1] for spec in melspecs) # 找到该批次中最长的时序长度

    # Pad each melspec to max_time_frames
    padded_melspecs = []
    for spec in melspecs:
        pad_width = max_time_frames - spec.shape[1] # 计算需要填充的宽度
        # Pad only the time dimension (axis 1)
        padded_spec = tf.pad(spec, [[0, 0], [0, pad_width]], mode='CONSTANT') # 仅在时间维度上填充
        padded_melspecs.append(padded_spec)

    # Stack the padded melspecs into a single tensor
    padded_melspecs = tf.stack(padded_melspecs, axis=0)  # Shape: (batch_size, c.NUM_MELS, max_time_frames)
    padded_melspecs = tf.expand_dims(padded_melspecs, axis=-1) # Add channel dimension # 添加通道维度，使其形状为 (batch_size, c.NUM_MELS, max_time_frames, c.CHANNELS)

    # Convert labels to a TensorFlow tensor
    labels = tf.stack(labels, axis=0) # 将标签堆叠成一个张量

    return padded_melspecs, labels # 返回填充后的 Mel 频谱图和标签

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
            melspec_path = os.path.join(melspec_dir, row['filename'] + '.npy') # 构建 Mel 频谱图的完整路径
            melspec = np.load(melspec_path) # 加载 Mel 频谱图
            label = label_map[row['speaker_id']] # 获取说话人对应的数字标签
            # Convert label to one-hot encoding *inside* the generator
            one_hot_label = tf.one_hot(label, depth=len(label_map)) # 将数字标签转换为 one-hot 编码
            yield melspec, one_hot_label # 产生 Mel 频谱图和 one-hot 编码的标签

    # Use tf.data.Dataset.from_generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(c.NUM_MELS, None), dtype=tf.float32),  # Dynamic time dimension # 指定 Mel 频谱图的形状，时间维度为 None，表示动态长度
            tf.TensorSpec(shape=(len(label_map),), dtype=tf.float32), #one hot # 指定标签的形状
        )
    )
    # Use padded_batch and set padding shape
    dataset = dataset.batch(batch_size).map(pad_and_stack) # Use the new pad_and_stack function # 使用 padded_batch 进行批处理和动态填充

    return dataset # 返回构建好的 TensorFlow Dataset

# ─── 训练模型函数 ───────────────────────────────────────────
def train_model(model, train_loader, test_loader):
    """
    训练模型。

    Args:
        model: 要训练的 Keras 模型。
        train_loader: 训练数据加载器（TensorFlow Dataset）。
        test_loader: 测试数据加载器（TensorFlow Dataset）。
    """
    steps = 0 # 初始化训练步数
    last_ckpt = get_last_checkpoint(c.PRE_CHECKPOINT_FOLDER) # 获取最后一个检查点
    if last_ckpt:
        model.load_weights(last_ckpt) # 如果存在检查点，则加载权重
        steps = int(last_ckpt.split('_')[-2]) # 从检查点文件名中提取步数

    while True: # 无限循环，直到达到所需的训练轮数或满足其他停止条件
        for x, y in train_loader: # Iterate through the tf.data.Dataset # 从训练数据加载器中获取一个批次的训练数据
            loss, acc = model.train_on_batch(x, y) # 在该批次上进行训练，并获取损失和准确率
            logging.info(f"Step {steps} — train loss={loss:.4f}, acc={acc:.4f}") # 打印训练信息

            if steps % c.TEST_PER_EPOCHS == 0: # 每 c.TEST_PER_EPOCHS 步进行一次测试
                for xt, yt in test_loader:
                    tl, ta = model.test_on_batch(xt, yt) # 在测试集上评估模型
                    logging.info(f"Step {steps} — test loss={tl:.4f}, acc={ta:.4f}") # 打印测试信息
                    break # Only test on the first batch for efficiency # 为了效率，仅在第一个测试批次上进行测试

            if steps % c.SAVE_PER_EPOCHS == 0: # 每 c.SAVE_PER_EPOCHS 步保存一次模型
                clean_old_checkpoints(c.PRE_CHECKPOINT_FOLDER, keep_latest=3) # 清理旧的检查点，仅保留最新的 3 个
                model.save_weights(
                    os.path.join(c.PRE_CHECKPOINT_FOLDER, f"model_{steps}_{loss:.4f}.h5") # 保存模型权重
                )
            steps += 1 # 增加训练步数

# ─── 主入口 ───────────────────────────────────────────
def main():
    """
    主函数，用于执行整个训练流程。
    """
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
    train_model(model, train_loader, test_loader) # 训练模型

if __name__ == "__main__":
    main() # 执行主函数
