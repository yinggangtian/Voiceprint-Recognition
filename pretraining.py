#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import numpy as np
import multiprocessing

import constants as c

# ─── CPU性能优化 - 自动检测CPU核心并设置环境变量 ───
# 在导入TensorFlow前设置CPU配置，只有在这里设置才会生效
cpu_count = multiprocessing.cpu_count()
print(f"检测到 {cpu_count} 个CPU核心，配置优化参数...")

# 为了最大化CPU使用率，设置更激进的线程配置
os.environ["OMP_NUM_THREADS"] = str(cpu_count)
os.environ["MKL_NUM_THREADS"] = str(cpu_count)
os.environ["KMP_BLOCKTIME"] = "0"  # 减少线程在空闲时的等待时间
os.environ["KMP_SETTINGS"] = "1"   # 显示KMP设置
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"  # 优化线程亲和性

# 优化TensorFlow线程设置 - 更激进地设置线程数以最大化CPU利用率
os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpu_count * 2)  # 增加线程数
os.environ["TF_NUM_INTEROP_THREADS"] = str(cpu_count)  # 增加并行操作数
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # 启用OneDNN优化
os.environ["TF_CPU_MULTI_WORKER_MIRRORED_STRATEGY"] = "1"  # 多工作器镜像策略
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 在TF加载前屏蔽大部分日志
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 避免TF预分配全部GPU内存

# 启用XLA JIT编译以提高性能 - 这是提高CPU利用率的关键
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"  # 全局启用XLA JIT优化

# 监控系统资源
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("提示: 安装psutil可监控系统资源使用情况 (pip install psutil)")

print(f"已自动配置CPU优化：检测到{cpu_count}个CPU核心")

# ─── 在设置环境变量后，再导入TensorFlow ───
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from time import time

from utils import load_metadata, build_label_map, split_metadata, get_last_checkpoint, clean_old_checkpoints
from models import convolutional_model
from gpu_utils import configure_gpu, print_gpu_info

# ─── 在任何 TensorFlow/Keras 导入之前，先屏蔽大部分日志 ───
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── 配置 GPU ─────────────────────────────────────────────────
gpus = configure_gpu()
print_gpu_info()

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
    # 配置TensorFlow使用所有可用的CPU进行训练
    # 设置线程优化策略
    print("配置模型以最大化CPU利用率...")
    # 确保使用XLA优化
    tf.config.optimizer.set_jit(True)  # 启用XLA JIT
    
    # 设置TensorFlow内存增长
    tf.config.experimental.set_memory_growth_for_gpu = False
    
    # 设置线程和GPU内存优化
    tf_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=cpu_count * 2,    # 增加内部操作并行度
        inter_op_parallelism_threads=cpu_count,        # 设置操作间并行度
        allow_soft_placement=True,                     # 允许TF选择更优的设备
        device_count={'CPU': cpu_count},               # 告知TF有多少CPU可用
        log_device_placement=False                     # 不记录设备放置日志，提高性能
    )
    # 启用全局XLA优化
    tf_config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    
    session = tf.compat.v1.Session(config=tf_config)
    tf.compat.v1.keras.backend.set_session(session)
    
    # 使用卷积模型作为基础
    base = convolutional_model(input_shape=input_shape) 
    
    # 添加Dropout层来防止过拟合
    x = tf.keras.layers.Dropout(0.5)(base.output)
    
    # 添加全连接层和softmax激活，使用L2正则化防止过拟合
    x = Dense(
        no_of_speakers, 
        activation='softmax', 
        name='softmax_layer',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)  # 添加L2正则化
    )(x)
    
    # 定义模型输入输出
    model = Model(inputs=base.input, outputs=x)
    
    # 使用Adam优化器和交叉熵损失，配置更高的优化器线程
    # 使用更高的epsilon值以增加数值稳定性，有助于在多线程环境中提高性能
    optimizer = Adam(learning_rate=0.001, epsilon=1e-7, beta_1=0.9, beta_2=0.999)
    
    # 尝试使用性能优化策略
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
    
    # 在CPU上使用合适的优化策略
    try:
        # 对TF 2.4+启用混合精度
        if tf_version >= (2, 4) and hasattr(tf.keras, 'mixed_precision'):
            print("启用混合精度训练 (mixed_float16)...")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            # 使用损失缩放来避免数值下溢
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("混合精度训练已启用，CPU性能应该显著提升")
        else:
            print("当前TensorFlow版本不支持CPU混合精度，使用标准精度")
    except Exception as e:
        print(f"优化策略未完全启用: {e}，使用标准配置")
    
    # 使用JIT编译优化模型编译过程 - 直接使用jit_compile参数
    # 避免使用 tf.xla.experimental.jit_scope() 因为它在Eager模式下不兼容
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy'],
        # 直接启用JIT编译，这在Eager模式下是兼容的
        jit_compile=True
    )
    
    # 打印模型结构
    logging.info(model.summary())
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
def paths_to_loaders(metadata, melspec_dir, label_map, batch_size, parallel_calls=None):
    """
    从元数据创建 TensorFlow Dataset。

    Args:
        metadata: 包含文件路径和说话人 ID 的 pandas DataFrame。
        melspec_dir: Mel 频谱图文件存储的目录。
        label_map: 将说话人 ID 映射到数字标签的字典。
        batch_size: 所需的批次大小。
        parallel_calls: 数据加载的并行度。默认为CPU核心数。

    Returns:
        一个 TensorFlow Dataset，它产生 (padded_melspecs, labels) 批次。
    """
    # 默认并行度为CPU核心数的2倍（因为IO操作可以并行更多）
    if parallel_calls is None:
        parallel_calls = cpu_count * 2
    
    # 使用多进程预加载数据
    print(f"开始预加载数据，并行度: {parallel_calls}")
    
    def generator():
        """
        生成器函数，用于按需加载 Mel 频谱图和标签。
        """
        # 预先创建文件路径列表，避免DataFrame迭代的开销
        all_paths = []
        all_labels = []
        
        for index, row in metadata.iterrows():
            try:
                speaker_id = str(row['speaker_id'])
                
                if 'file_path' in row:
                    file_path = row['file_path']
                    file_name = os.path.basename(file_path)
                    file_name_no_ext = os.path.splitext(file_name)[0]
                    melspec_path = os.path.join(melspec_dir, file_name_no_ext + '.npy')
                    
                    if not os.path.exists(melspec_path) and 'subset' in row:
                        subset = row['subset']
                        melspec_path = os.path.join(melspec_dir, subset, f"{speaker_id}_{file_name_no_ext}.npy")
                else:
                    melspec_path = os.path.join(melspec_dir, row.get('filename', '') + '.npy')
                
                if os.path.exists(melspec_path):
                    all_paths.append(melspec_path)
                    all_labels.append(label_map[row['speaker_id']])
            except (FileNotFoundError, KeyError) as e:
                logging.warning(f"跳过文件 {melspec_path if 'melspec_path' in locals() else '未知路径'}: {e}")
        
        # 随机打乱文件和标签
        indices = list(range(len(all_paths)))
        np.random.shuffle(indices)
        
        for idx in indices:
            try:
                # 加载梅尔频谱图
                melspec = np.load(all_paths[idx])
                # 将数字标签转换为 one-hot 编码
                one_hot_label = tf.one_hot(all_labels[idx], depth=len(label_map))
                # 产生 Mel 频谱图和 one-hot 编码的标签
                yield melspec, one_hot_label
            except Exception as e:
                logging.warning(f"加载失败: {all_paths[idx]} - {e}")

    # Use tf.data.Dataset.from_generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(c.NUM_MELS, None), dtype=tf.float32),  # Dynamic time dimension
            tf.TensorSpec(shape=(len(label_map),), dtype=tf.float32),   # one-hot labels
        )
    )
    
    # 设置更大的缓冲区和并行度以充分利用CPU
    buffer_size = min(50000, len(metadata) * 20)  # 设置一个更大的缓冲区
    
    # 优化数据管道 - 使用更积极的并行加载和预取
    print(f"设置数据管道: 缓冲区大小={buffer_size}, 批次大小={batch_size}, 并行调用数={parallel_calls}")
    
    # 优化数据管道 - 更积极的并行和预取策略
    dataset = dataset.cache()  # 缓存数据以加快重复的epoch
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)  # 更大的缓冲区随机打乱数据
    dataset = dataset.padded_batch(
        batch_size, 
        padded_shapes=((c.NUM_MELS, None), (len(label_map),)),
        padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
    )
    dataset = dataset.map(
        pad_and_stack, 
        num_parallel_calls=tf.data.AUTOTUNE  # 使用自动调整的并行度来最大化利用CPU
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 预取数据
    
    return dataset

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
    # 设置CPU优先级（如果可能）
    try:
        import psutil
        process = psutil.Process(os.getpid())
        if sys.platform == 'win32':
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            # 在Linux/Mac上设置较高的优先级
            os.nice(-10)  # 尝试设置较高的优先级
        print("已提高CPU优先级以最大化CPU资源利用")
    except (ImportError, OSError, psutil.AccessDenied):
        print("无法提高进程优先级，使用默认优先级")
                
    steps = 0 # 初始化训练步数
    last_ckpt = get_last_checkpoint(c.PRE_CHECKPOINT_FOLDER) # 获取最后一个检查点
    if last_ckpt:
        print(f"加载检查点: {last_ckpt}")
        model.load_weights(last_ckpt) # 如果存在检查点，则加载权重
        try:
            # 从检查点文件名中提取步数
            steps = int(last_ckpt.split('_')[-2]) 
            print(f"从步骤 {steps} 继续训练")
        except (ValueError, IndexError):
            print("无法从检查点获取步数，从0开始")
            steps = 0

    start_time = time()
    
    # 预取并优化数据加载
    # 使用更大的预取缓冲区以确保CPU始终有任务处理
    buffer_multiplier = max(1, min(4, cpu_count // 4))  # 根据CPU核心数调整缓冲区大小
    train_loader = train_loader.prefetch(buffer_multiplier * tf.data.AUTOTUNE)
    test_loader = test_loader.prefetch(buffer_multiplier * tf.data.AUTOTUNE)
    
    print(f"数据加载优化: 预取缓冲区大小 = {buffer_multiplier}×AUTOTUNE")
    
    # 初始化迭代器
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    
    # 使用XLA编译优化训练步骤函数 - 这将帮助更好地利用多核CPU
    # experimental_compile=True在较低版本TF中等同于jit_compile=True
    @tf.function(experimental_compile=True)
    def train_step(x, y):
        """使用XLA编译的训练步骤函数 - 提高CPU利用率"""
        return model.train_on_batch(x, y)
    
    @tf.function(jit_compile=True)
    def test_step(x, y):
        """使用XLA编译的测试步骤函数"""
        return model.test_on_batch(x, y)
    
    # 定义资源统计函数
    def get_resource_usage():
        """获取当前系统资源使用情况并提供优化建议"""
        if not HAS_PSUTIL:
            return "资源监控未启用(需安装psutil)"
        
        # 获取CPU使用率 - 使用全局所有CPU的平均值
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=False)
        # 获取每个CPU核心的使用率
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        # 计算CPU核心利用的标准差 - 用于检测负载不均衡
        cpu_std = np.std(per_cpu) if len(per_cpu) > 1 else 0
        
        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)
        
        # 检测CPU利用率问题并提供建议
        cpu_warning = ""
        if cpu_percent < 30:
            # 严重的CPU利用率不足
            cpu_warning = " [CPU利用率过低!]"
            # 每25步提供详细建议
            if steps % 25 == 0:
                print("\n[CPU优化建议] CPU利用率过低，考虑:")
                print("1. 增加批次大小 (当前批次大小可能太小)")
                print("2. 设置TF_XLA_FLAGS=\"--tf_xla_auto_jit=2\"环境变量") 
                print("3. 检查数据加载是否成为瓶颈\n")
        elif cpu_percent < 60:
            # 中等CPU利用率不足
            cpu_warning = " [CPU利用率偏低]"
            
        # 检测内存压力
        mem_warning = ""
        if memory_percent > 90:
            mem_warning = " [内存不足!]"
            
        # 检测CPU负载不均衡
        balance_info = ""
        if cpu_std > 25 and cpu_percent > 30:  # 只在总体利用率合理但分布不均匀时警告
            balance_info = f" [CPU负载不均衡, 标准差={cpu_std:.1f}]"
            
        # 构造详细的资源使用报告
        detailed = f"CPU: {cpu_percent:5.1f}%{cpu_warning}{balance_info} | 内存: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory_percent:.1f}%){mem_warning}"
        return detailed

    print(f"\n开始训练 (XLA优化已启用) - 最大步数: {max_steps if max_steps else '无限'}\n")
    
    while True: # 无限循环，直到达到所需的训练轮数或满足其他停止条件
        try:
            # 尝试获取下一批数据，如果迭代器耗尽则重新创建
            try:
                x, y = next(train_iter)
            except (StopIteration, tf.errors.OutOfRangeError):
                print("\n训练数据迭代器重置")
                train_iter = iter(train_loader.prefetch(tf.data.AUTOTUNE))
                x, y = next(train_iter)
            
            # 使用编译后的训练步骤进行训练 - 应该有更高的CPU利用率
            loss, acc = train_step(x, y)
            
            # 使用精简格式输出训练信息（单行输出）
            resource_status = get_resource_usage()
            
            if max_steps is not None:
                progress_percent = (steps / max_steps) * 100
                elapsed_time = time() - start_time
                estimated_total_time = elapsed_time / (steps + 1) * max_steps
                estimated_remaining_time = estimated_total_time - elapsed_time
                
                print(f"\r【步骤 {steps:4d}/{max_steps} ({progress_percent:5.1f}%)】" +
                      f"损失: {loss:.4f} | 准确率: {acc:.4f} | " +
                      f"剩余: {estimated_remaining_time:.1f}秒 | {resource_status}", end="")
                
                # 每10步换一次行，保持输出整洁
                if steps % 10 == 0:
                    print("")
                    
                logging.info(f"Step {steps}/{max_steps} ({progress_percent:.1f}%) — loss={loss:.4f}, acc={acc:.4f} — Est. remaining: {estimated_remaining_time:.2f}s")
            else:
                print(f"\r【步骤 {steps:4d}】损失: {loss:.4f} | 准确率: {acc:.4f} | {resource_status}", end="")
                if steps % 10 == 0:
                    print("")
                logging.info(f"Step {steps} — loss={loss:.4f}, acc={acc:.4f}")

            if steps % c.TEST_PER_EPOCHS == 0: # 每 c.TEST_PER_EPOCHS 步进行一次测试
                try:
                    xt, yt = next(test_iter)
                except (StopIteration, tf.errors.OutOfRangeError):
                    print("\n测试数据迭代器重置")
                    test_iter = iter(test_loader.prefetch(tf.data.AUTOTUNE))
                    xt, yt = next(test_iter)
                
                # 使用编译后的测试步骤函数
                tl, ta = test_step(xt, yt)
                
                # 使用精简的单行格式显示测试结果
                print(f"\n【测试 - 步骤 {steps}】测试损失: {tl:.4f} | 测试准确率: {ta:.4f} | {resource_status}")
                
                logging.info(f"Step {steps} — test loss={tl:.4f}, acc={ta:.4f}")

            if steps % c.SAVE_PER_EPOCHS == 0: # 每 c.SAVE_PER_EPOCHS 步保存一次模型
                clean_old_checkpoints(c.PRE_CHECKPOINT_FOLDER, keep_latest=3) # 清理旧的检查点，仅保留最新的 3 个
                model.save_weights(
                    os.path.join(c.PRE_CHECKPOINT_FOLDER, f"model_{steps}_{loss:.4f}.weights.h5")
                )
                print(f"\n保存检查点: model_{steps}_{loss:.4f}.weights.h5")
            
            steps += 1 # 增加训练步数
            
            # 检查是否达到最大步数
            if max_steps is not None and steps >= max_steps:
                print("\n")  # 确保最终输出在新行上
                logging.info(f"已达到最大步数 ({max_steps})。训练完成。")
                return
                
        except KeyboardInterrupt:
            print("\n\n训练被用户中断")
            return

# ─── 主入口 ───────────────────────────────────────────
def main():
    """
    主函数，用于执行整个训练流程。
    """
    # 创建必要的目录
    os.makedirs(c.PRE_CHECKPOINT_FOLDER, exist_ok=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='预训练声纹识别模型。')
    parser.add_argument('--max_steps', type=int, default=10000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=None, help='训练批次大小，越大越能充分利用CPU')
    parser.add_argument('--parallel_calls', type=int, default=None, help='数据加载并行数量')
    args = parser.parse_args()
    
    # 智能设置批次大小 - 根据可用内存自动调整
    if args.batch_size is None:
        # 根据CPU数和可用内存计算适当的批次大小
        if HAS_PSUTIL:
            # 获取系统内存信息（GB）
            available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            # 根据经验公式计算批次大小 (每8GB内存大约支持128的批次大小)
            recommended_batch_size = min(512, max(32, int(available_memory_gb * 16)))
            # 确保批次大小是32的倍数
            batch_size = (recommended_batch_size // 32) * 32
            print(f"根据系统可用内存 ({available_memory_gb:.1f}GB) 自动设置批次大小: {batch_size}")
        else:
            # 如果没有psutil，设置一个合理的默认值
            batch_size = 128 * (cpu_count // 4)  # 每4个CPU核心设置128的批次
            batch_size = max(64, min(512, batch_size))  # 限制在64-512之间
            print(f"自动设置批次大小为: {batch_size} (根据CPU核心数)")
    else:
        batch_size = args.batch_size
        
    # 设置并行度 - 更高的并行度有助于提高CPU利用率
    if args.parallel_calls is None:
        # 默认并行度设为CPU核心数的2倍
        parallel_calls = cpu_count * 2
    else:
        parallel_calls = args.parallel_calls
        
    print(f"优化配置: 批次大小={batch_size}, 并行度={parallel_calls}, XLA优化=已启用")
    
    # 配置日志记录
    logging.basicConfig(
        handlers=[logging.StreamHandler(sys.stdout)], # 将日志输出到标准输出
        level=logging.INFO, # 设置日志级别为 INFO
        format="%(asctime)s [%(levelname)s] %(message)s" # 设置日志格式
    )

    meta = load_metadata("metadata_small.csv") # 加载元数据
    train_meta, test_meta = split_metadata(meta, train_frac=0.8) # 将元数据分割为训练集和测试集

    # 打印数据集大小信息
    print(f"训练集大小: {len(train_meta)}条数据, 测试集大小: {len(test_meta)}条数据")
    
    label_map = build_label_map(meta) # 构建标签映射，将说话人 ID 映射到数字标签
    print(f"总说话人数: {len(label_map)}")
    
    # 使用命令行指定的批次大小创建数据加载器
    train_loader = paths_to_loaders(train_meta, "melspec_small", label_map, batch_size, parallel_calls)
    test_loader = paths_to_loaders(test_meta, "melspec_small", label_map, batch_size, parallel_calls)

    # 新的输入形状
    input_shape = (c.NUM_MELS, None, c.CHANNELS)
    model = initialize_model(input_shape, no_of_speakers=len(label_map)) # 初始化模型
    train_model(model, train_loader, test_loader, max_steps=args.max_steps) # 训练模型

if __name__ == "__main__":
    main() # 执行主函数
