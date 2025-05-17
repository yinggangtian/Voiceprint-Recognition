#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train voiceprint recognition with CNN and optional GRU models.
Optimized for maximum CPU utilization with parallel processing and XLA.
"""
import logging
import argparse
import sys
import os
import traceback
from time import time, sleep
import multiprocessing
import numpy as np

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

# 在设置环境变量后，再导入TensorFlow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Project imports
from gpu_utils import configure_gpu, print_gpu_info
import constants as c
from pre_process import data_catalog, preprocess_and_save
from models import convolutional_model, recurrent_model, recurrent_model_softmax
from random_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss, softmax_loss
from utils import get_last_checkpoint, create_dir_and_delete_content
from test_model import eval_model
import select_batch

# 资源使用监控函数
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

# Ensure necessary directories exist
def prepare_dirs():
    dirs = [c.CHECKPOINT_FOLDER,
            c.BEST_CHECKPOINT_FOLDER,
            c.GRU_CHECKPOINT_FOLDER,
            os.path.dirname(c.LOSS_LOG),
            os.path.dirname(c.TEST_LOG)]
    for d in dirs:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

# 优化模型编译和训练
def optimize_model(model, loss_fn=None):
    """
    使用XLA JIT编译和其他优化策略优化模型，提高CPU利用率
    """
    # 配置TensorFlow使用所有可用的CPU进行训练
    print("配置模型以最大化CPU利用率...")
    
    # 确保使用XLA优化
    tf.config.optimizer.set_jit(True)  # 启用XLA JIT
    
    # 设置更激进的线程和优化配置
    tf_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=cpu_count * 2,    # 增加内部操作并行度
        inter_op_parallelism_threads=cpu_count,        # 设置操作间并行度
        allow_soft_placement=True,                     # 允许TF选择更优的设备
        device_count={'CPU': cpu_count},               # 告知TF有多少CPU可用
        log_device_placement=False                     # 不记录设备放置日志，提高性能
    )
    # 启用全局XLA优化
    tf_config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    
    # 启用内存优化 - 仅对GPU设备设置memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"已为GPU设备启用memory growth: {device}")
            except Exception as e:
                print(f"无法为设备启用memory growth: {device}, 错误: {e}")
    
    # 设置TensorFlow会话
    session = tf.compat.v1.Session(config=tf_config)
    tf.compat.v1.keras.backend.set_session(session)
    
    # 配置高性能优化器设置
    optimizer = Adam(
        learning_rate=0.001, 
        epsilon=1e-7, 
        beta_1=0.9, 
        beta_2=0.999,
        amsgrad=True  # 使用AMSGrad变体，可能提高性能
    )
    
    # 只在有GPU的情况下启用混合精度训练
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
    has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    
    if has_gpu and tf_version >= (2, 4) and hasattr(tf.keras, 'mixed_precision'):
        try:
            print("检测到GPU，启用混合精度训练以提高性能...")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("混合精度训练已启用")
        except Exception as e:
            print(f"混合精度训练未启用: {e}")
    else:
        if not has_gpu:
            print("未检测到GPU，禁用混合精度训练")
        # 确保在CPU模式下使用float32精度
        if hasattr(tf.keras, 'mixed_precision'):
            tf.keras.mixed_precision.set_global_policy('float32')
    
    # 使用JIT编译优化模型编译过程
    if loss_fn is None:
        loss_fn = deep_speaker_loss
    
    # 创建训练步骤函数 - 改进以处理XLA编译
    @tf.function(jit_compile=True)
    def train_fn(x, y):
        """被JIT编译优化的训练函数"""
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = loss_fn(y, y_pred)
        
        # 获取梯度并应用到模型 - 使用try/except捕获形状不匹配错误
        try:
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            print(f"梯度计算错误: {e}")
            # 不更新参数，但返回损失值
            return loss_value
            
        return loss_value
    
    # 让模型知道如何使用优化的训练函数
    model.train_fn = train_fn
    
    # 编译模型
    model.compile(
        optimizer=optimizer, 
        loss=loss_fn,
        # 在Keras编译中直接启用jit_compile，而不使用jit_scope
        jit_compile=True
    )
    
    return model

# 创建训练步骤函数
def create_train_step(model):
    """
    创建训练步骤函数，确保返回的是Python标量值而不是Tensor
    
    Args:
        model: 要训练的模型
        
    Returns:
        train_step函数，该函数接受(x,y)并返回损失值
    """
    def train_step(x, y):
        # 确保批次大小是3的倍数，这对于triplet_loss是必须的
        batch_size = x.shape[0]
        if batch_size % 3 != 0:
            logging.warning(f"训练批次大小 {batch_size} 不是3的倍数，将被调整")
            adjust_to = (batch_size // 3) * 3
            x = x[:adjust_to]
            # 标签只对应anchor部分
            anchor_count = adjust_to // 3
            if len(y) > anchor_count:
                y = y[:anchor_count]
        
        try:
            # train_on_batch返回Tensor或标量，我们确保它是Python标量
            loss = model.train_on_batch(x, y)
            # 如果loss是Tensor，转换为Python标量
            if hasattr(loss, 'numpy'):
                return float(loss.numpy())
            return float(loss)
        except tf.errors.InvalidArgumentError as e:
            # 捕获形状不匹配等错误
            logging.error(f"训练错误: {e}")
            return 999.0  # 返回一个明显异常的值表示错误
    
    return train_step

# 用于自适应批次大小的函数
def build_and_train_models(libri, speakers, spk_index, batch_size, candidates_per_batch, max_steps=None, current_step=0):
    """
    构建模型并执行训练循环。该函数作为独立单元，以便在OOM错误后可以重建模型。
    
    Args:
        libri: 数据集信息
        speakers: 说话人列表
        spk_index: 说话人到索引的映射
        batch_size: 批次大小
        candidates_per_batch: 每批次候选项数量
        max_steps: 最大训练步数
        current_step: 当前训练步数
    
    Returns:
        训练步数
    """
    logging.info(f"开始构建模型 (批次大小={batch_size}, 候选数={candidates_per_batch})")
    
    # 保存当前配置到全局变量
    c.BATCH_SIZE = batch_size
    c.CANDIDATES_PER_BATCH = candidates_per_batch
    
    # 计算总批次大小
    batch_size_total = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    
    # 清理TensorFlow会话，避免内存泄漏
    tf.keras.backend.clear_session()
    
    # 为了确定模型输入形状，构建一个示例批次
    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=speakers)
    x0, y0 = batch.to_inputs()
    num_frames = x0[0].shape[0]
    input_shape = (num_frames, x0[0].shape[1], x0[0].shape[2])
    logging.info(f"输入形状: {input_shape}, 总批次大小: {batch_size_total}")
    
    # 构建CNN模型
    model = convolutional_model(input_shape=input_shape)
    model = optimize_model(model, deep_speaker_loss)
    model.summary(print_fn=lambda s: logging.info(s))
    
    # 加载CNN检查点
    cp = None
    if c.PRE_TRAIN:
        cp = get_last_checkpoint(c.PRE_CHECKPOINT_FOLDER)
        if cp:
            logging.info(f"加载预训练检查点: {cp}")
            x = Dense(len(speakers), activation='softmax')(model.output)
            pre_model = Model(model.input, x)
            pre_model.load_weights(cp)
    else:
        cp = get_last_checkpoint(c.CHECKPOINT_FOLDER)
        if cp:
            model.load_weights(cp)
            logging.info(f"加载模型检查点: {cp}")
    
    # 构建GRU模型（如果启用）
    gru_model = None
    if c.COMBINE_MODEL:
        if c.use_softmax_loss:
            gru_model = recurrent_model_softmax(
                input_shape=input_shape,
                batch_size=batch_size_total,
                num_frames=num_frames,
                num_spks=len(speakers)
            )
            gru_model = optimize_model(gru_model, softmax_loss(len(speakers)))
        else:
            gru_model = recurrent_model(
                input_shape=input_shape,
                batch_size=batch_size_total,
                num_frames=num_frames
            )
            gru_model = optimize_model(gru_model, deep_speaker_loss)
        
        gru_model.summary(print_fn=lambda s: logging.info(s))
        
        # 加载GRU检查点
        if not c.PRE_TRAIN:
            cp2 = get_last_checkpoint(c.GRU_CHECKPOINT_FOLDER)
            if cp2:
                gru_model.load_weights(cp2)
                logging.info(f"加载GRU检查点: {cp2}")
    
    # 创建训练步骤函数
    cnn_train_step = create_train_step(model)
    gru_train_step = create_train_step(gru_model) if c.COMBINE_MODEL else None
    
    # 训练循环
    grad_steps = current_step
    start = time()
    
    while True:
        try:
            # 选择最佳批次
            t0 = time()
            x_batch, y_ids = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
            
            # 确保批次大小是3的倍数
            if x_batch.shape[0] % 3 != 0:
                adjust_to = (x_batch.shape[0] // 3) * 3
                logging.warning(f"批次大小 {x_batch.shape[0]} 不是3的倍数，调整为 {adjust_to}")
                x_batch = x_batch[:adjust_to]
                if len(y_ids) > adjust_to // 3:
                    y_ids = y_ids[:adjust_to // 3]
            
            # 转换标签到索引
            y_true = np.array([spk_index[i] for i in y_ids])
            
            # 对于三元组损失，我们总是需要3倍的标签（每个锚点对应一个三元组）
            # 首先确保我们有正确数量的锚点标签
            anchor_count = x_batch.shape[0] // 3
            logging.info(f"批次结构: 输入: {x_batch.shape[0]}个样本 (应该是{anchor_count}个三元组)")
            
            # 确保有正确数量的标签
            if len(y_true) > anchor_count:
                logging.warning(f"标签过多，裁剪: {len(y_true)} -> {anchor_count}")
                y_true = y_true[:anchor_count]
            elif len(y_true) < anchor_count:
                logging.warning(f"标签不足，填充: {len(y_true)} -> {anchor_count}")
                # 如果标签不足，复制最后一个标签来填充
                y_true = np.pad(y_true, (0, anchor_count - len(y_true)), mode='edge')
                
            # 重复标签以匹配三元组结构（为每个三元组重复标签）
            logging.info(f"重复标签以匹配三元组结构: {len(y_true)} -> {len(y_true)*3}")
            y_true = np.repeat(y_true, 3)  # 形状 (3N,)
            
            # 最终检查确保输入和标签形状匹配
            if len(y_true) != x_batch.shape[0]:
                logging.warning(f"标签数量调整 {len(y_true)} -> {x_batch.shape[0]}")
                # 确保标签和输入数量匹配
                if len(y_true) < x_batch.shape[0]:
                    # 如果标签少于输入，复制最后一个标签直到匹配
                    y_true = np.pad(y_true, (0, x_batch.shape[0] - len(y_true)), 'edge')
                else:
                    # 如果标签多于输入，截断
                    y_true = y_true[:x_batch.shape[0]]
            
            logging.info(f"最终批次形状: x_batch={x_batch.shape}, y_true={y_true.shape}")
            batch_select_time = time()-t0
            
            # 资源使用监控
            resource_status = get_resource_usage()
            
            # 进度显示
            if max_steps:
                pct = grad_steps / max_steps * 100
                elapsed = time() - start
                rem = elapsed / (grad_steps+1-current_step) * (max_steps - grad_steps)
                logging.info(f"步骤 {grad_steps}/{max_steps} ({pct:.1f}%), 剩余 {rem:.1f}s, 批次时间={batch_select_time:.2f}s | {resource_status}")
            else:
                logging.info(f"步骤 {grad_steps}, 批次时间={batch_select_time:.2f}s | {resource_status}")
            
            # CNN训练
            loss = 0
            if not c.COMBINE_MODEL or (c.COMBINE_MODEL and c.CNN_MODEL_TRAIN):
                loss = cnn_train_step(x_batch, y_true)  # 返回的是 float 值
                logging.info(f"CNN损失 {loss:.4f}")
            
            # GRU训练
            loss1 = 0
            if c.COMBINE_MODEL:
                try:
                    loss1 = gru_train_step(x_batch, y_true)  # 返回的是 float 值
                    logging.info(f"GRU损失 {loss1:.4f}")
                    with open(os.path.join(c.GRU_CHECKPOINT_FOLDER, 'losses_gru.txt'), 'a') as f:
                        f.write(f"{grad_steps},{loss1}\n")
                except Exception as ex:
                    logging.error(f"GRU训练错误: {ex}")
            
            # 记录CNN损失
            with open(c.LOSS_LOG, 'a') as f:
                f.write(f"{grad_steps},{loss}\n")
            
            # 定期评估
            if grad_steps % 10 == 0:
                try:
                    # 保存当前模型以便评估
                    temp_model_path = f"{c.CHECKPOINT_FOLDER}/temp_model_{grad_steps}.h5"
                    model.save_weights(temp_model_path)
                    
                    # 评估并打印详细的性能指标
                    fm, tpr, acc, eer, frr, far = eval_model(
                        model,
                        train_batch_size=batch_size_total,
                        test_dir=c.TEST_DIR,
                        check_partial=True,
                        gru_model=gru_model if c.COMBINE_MODEL else None
                    )
                    
                    # 使用多个日志级别确保信息显示
                    detailed_metrics = f"评估结果 - EER: {eer:.4f}, 准确率: {acc:.4f}, F-measure: {fm:.4f}, TPR: {tpr:.4f}, FRR: {frr:.4f}, FAR: {far:.4f}"
                    print("\n" + "="*80)
                    print(detailed_metrics)
                    print("="*80 + "\n")
                    logging.info(detailed_metrics)
                    
                    # 记录到文件
                    with open(c.TEST_LOG, 'a') as f:
                        f.write(f"{grad_steps},{eer:.4f},{fm:.4f},{acc:.4f},{tpr:.4f},{frr:.4f},{far:.4f}\n")
                    
                    # 清理临时模型
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)
                        
                except Exception as eval_ex:
                    logging.error(f"评估错误: {eval_ex}")
                    traceback.print_exc()
            
            # 保存检查点
            if max_steps and grad_steps >= max_steps:
                # 最终保存模型权重
                model.save_weights(f"{c.CHECKPOINT_FOLDER}/model_final_{grad_steps}_{loss:.5f}.h5")
                if c.COMBINE_MODEL:
                    gru_model.save_weights(f"{c.GRU_CHECKPOINT_FOLDER}/grumodel_final_{grad_steps}_{loss1:.5f}.h5")
                break
            
            if grad_steps % c.SAVE_PER_EPOCHS == 0:
                create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
                model.save_weights(f"{c.CHECKPOINT_FOLDER}/model_{grad_steps}_{loss:.5f}.h5")
                if c.COMBINE_MODEL:
                    gru_model.save_weights(f"{c.GRU_CHECKPOINT_FOLDER}/grumodel_{grad_steps}_{loss1:.5f}.h5")
            
            grad_steps += 1
            
        except KeyboardInterrupt:
            print("\n\n训练被用户中断")
            # 保存中断时的模型
            model.save_weights(f"{c.CHECKPOINT_FOLDER}/model_interrupted_{grad_steps}_{loss:.5f}.h5")
            if c.COMBINE_MODEL:
                gru_model.save_weights(f"{c.GRU_CHECKPOINT_FOLDER}/grumodel_interrupted_{grad_steps}_{loss1:.5f}.h5")
            return grad_steps
        
        except tf.errors.ResourceExhaustedError as oom_error:
            # 发生OOM错误，保存当前进度，准备重新构建模型
            logging.error(f"内存不足 (OOM): {oom_error}")
            logging.info("保存当前模型状态并尝试减小批次大小...")
            
            try:
                # 尝试保存模型权重
                model.save_weights(f"{c.CHECKPOINT_FOLDER}/model_oom_{grad_steps}_{loss:.5f}.h5")
                if c.COMBINE_MODEL and gru_model is not None:
                    gru_model.save_weights(f"{c.GRU_CHECKPOINT_FOLDER}/grumodel_oom_{grad_steps}_{loss1:.5f}.h5")
            except Exception as save_ex:
                logging.error(f"保存OOM检查点失败: {save_ex}")
            
            # 抛出异常让外层函数处理批次大小调整
            raise
    
    return grad_steps

# Main function
def main(libri_dir=c.DATASET_DIR, max_steps=None, batch_size=None):
    prepare_dirs()
    
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

    # Configure GPU
    gpus = configure_gpu()
    print_gpu_info()

    # 智能设置批次大小 - 根据可用内存自动调整
    if batch_size is None:
        # 根据CPU数和可用内存计算适当的批次大小
        if HAS_PSUTIL:
            # 获取系统内存信息（GB）
            available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            # 根据经验公式计算批次大小 (每8GB内存大约支持128的批次大小)
            recommended_batch_size = min(32, max(4, int(available_memory_gb * 4)))
            batch_size = recommended_batch_size
            print(f"根据系统可用内存 ({available_memory_gb:.1f}GB) 自动设置批次大小: {batch_size}")
        else:
            # 如果没有psutil，设置一个合理的默认值
            batch_size = min(32, max(4, cpu_count // 2))  # 每2个CPU核心设置1个批次
            print(f"自动设置批次大小为: {batch_size} (根据CPU核心数)")
    else:
        batch_size = c.BATCH_SIZE
    
    # 自适应设置CANDIDATES_PER_BATCH
    candidates_per_batch = c.CANDIDATES_PER_BATCH
    
    # 如果没有明确设置，则根据批次大小按比例设置候选项数量
    # 保持与原始比例一致: CANDIDATES_PER_BATCH / BATCH_SIZE
    original_ratio = c.CANDIDATES_PER_BATCH / c.BATCH_SIZE  # 原始比例
    candidates_per_batch = int(batch_size * original_ratio)
    
    # 确保候选项数量在合理范围内
    candidates_per_batch = max(batch_size * 2, min(candidates_per_batch, 2000))
    
    print(f"使用批次大小: {batch_size}, 候选项数量: {candidates_per_batch}")
    
    # 设置全局批次大小
    c.BATCH_SIZE = batch_size
    c.CANDIDATES_PER_BATCH = candidates_per_batch

    # Load or preprocess data
    logging.info(f"正在 {libri_dir} 中查找特征文件")
    libri = data_catalog(libri_dir, pattern='**/*.npy')
    if libri is None or len(libri) == 0:
        logging.warning("未找到.npy文件，正在运行预处理...")
        # create dataset dirs
        os.makedirs(c.DATASET_DIR, exist_ok=True)
        os.makedirs(os.path.join(c.DATASET_DIR, 'train-clean-100'), exist_ok=True)
        os.makedirs(os.path.join(c.DATASET_DIR, 'test-clean'), exist_ok=True)
        preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        libri = data_catalog(libri_dir, pattern='**/*.npy')
        if libri is None or len(libri) == 0:
            logging.error("预处理失败，未找到数据。退出。")
            sys.exit(1)

    # Prepare speakers dictionary
    speakers = np.sort(libri['speaker_id'].unique())
    spk_index = {spk: idx for idx, spk in enumerate(speakers)}
    # Map speaker to file list
    spk_utt = {spk: [] for spk in speakers}
    for fn, lbl in zip(libri['filename'], libri['speaker_id']):
        spk_utt[lbl].append(fn)
    # Filter speakers with <2 utterances
    spk_utt = {spk: utts for spk, utts in spk_utt.items() if len(utts) > 1}
    speakers = list(spk_utt.keys())
    select_batch.create_data_producer(speakers, spk_utt)

    # 实现自适应训练过程，带有自动批次大小调整
    grad_steps = 0
    retry_count = 0
    max_retries = getattr(c, 'MAX_RETRIES', 5)
    min_batch_size = getattr(c, 'MIN_BATCH_SIZE', 2)  # 最小可接受的批次大小
    oom_recovery_factor = getattr(c, 'OOM_RECOVERY_FACTOR', 0.6)  # OOM后减少到原来的60%
    
    # 显示训练配置信息
    print("=" * 50)
    print(f"训练配置:")
    print(f"- 初始批次大小: {batch_size}")
    print(f"- 候选项数量: {candidates_per_batch}")
    print(f"- 最小批次大小: {min_batch_size}")
    print(f"- 最大重试次数: {max_retries}")
    print(f"- OOM恢复系数: {oom_recovery_factor}")
    print(f"- 填充模式: {getattr(c, 'PAD_MODE', 'none')}")
    print(f"- CPU核心数: {cpu_count}")
    print("=" * 50)
    
    while retry_count < max_retries and batch_size >= min_batch_size:
        try:
            logging.info(f"尝试使用批次大小={batch_size}, 候选项数量={candidates_per_batch} 训练模型")
            
            # 调用自适应训练函数，这个函数会处理模型构建和训练循环
            # 返回值是当前的训练步数
            grad_steps = build_and_train_models(
                libri=libri,
                speakers=speakers,
                spk_index=spk_index,
                batch_size=batch_size,
                candidates_per_batch=candidates_per_batch,
                max_steps=max_steps,
                current_step=grad_steps
            )
            
            # 如果训练完成，退出循环
            break
            
        except tf.errors.ResourceExhaustedError as oom_error:
            retry_count += 1
            logging.error(f"内存不足错误 (OOM) (尝试 {retry_count}/{max_retries})")
            logging.error(f"错误信息: {oom_error}")
            
            # 显示当前内存使用情况
            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                logging.info(f"当前内存使用: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")
            
            # 计算新的批次大小和候选项数量
            old_batch_size = batch_size
            old_candidates = candidates_per_batch
            
            # 减少批次大小和候选项数量
            batch_size = max(min_batch_size, int(batch_size * oom_recovery_factor))
            candidates_per_batch = max(batch_size * 2, int(candidates_per_batch * oom_recovery_factor))
            
            logging.info(f"减小批次大小: {old_batch_size} -> {batch_size}")
            logging.info(f"减小候选项数: {old_candidates} -> {candidates_per_batch}")
            
            # 更新全局变量
            c.BATCH_SIZE = batch_size
            c.CANDIDATES_PER_BATCH = candidates_per_batch
            
            # 清理内存
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            # 在重试前短暂暂停，让系统释放资源
            logging.info("暂停5秒钟，等待内存释放...")
            sleep(5)
        
        except KeyboardInterrupt:
            logging.info("训练被用户中断")
            return
        
        except Exception as ex:
            logging.error(f"训练过程中发生未知错误: {ex}")
            import traceback
            traceback.print_exc()
            # 如果是其他错误，也尝试减小批次大小重试一次
            if retry_count < 1:  # 只对其他错误重试一次
                retry_count += 1
                old_batch_size = batch_size
                batch_size = max(min_batch_size, int(batch_size * 0.8))  # 减少到80%
                logging.info(f"减小批次大小: {old_batch_size} -> {batch_size}，尝试恢复...")
                c.BATCH_SIZE = batch_size
                sleep(2)
            else:
                raise  # 重试后仍然失败，抛出异常
    
    if retry_count >= max_retries:
        logging.error(f"在 {max_retries} 次尝试后仍然发生OOM错误，已达到最小批次大小 {min_batch_size}。")
        logging.error("建议：")
        logging.error("1. 尝试减少模型大小或复杂度")
        logging.error("2. 增加系统内存")
        logging.error("3. 尝试使用fixed-length模式 (在constants.py中设置NUM_FRAMES为固定值)")
        logging.error("4. 修改PAD_MODE (在constants.py中设置，可选'zero'或'repeat')")
    else:
        logging.info(f"训练完成，共执行 {grad_steps} 步")
        if retry_count > 0:
            logging.info(f"最终批次大小: {batch_size}, 候选项数量: {candidates_per_batch}")
            logging.info(f"(原始值分别为: {args.batch_size if args.batch_size else 'auto'}, {c.CANDIDATES_PER_BATCH})")

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description='训练声纹识别模型 (CPU优化版)')
    parser.add_argument('--max_steps', type=int, default=None, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=None, help='训练批次大小，越大越能充分利用CPU')
    parser.add_argument('--candidates', type=int, default=None, help='每批次候选项数量')
    args = parser.parse_args()
    
    # 添加对候选项数量的命令行参数支持
    if args.candidates:
        c.CANDIDATES_PER_BATCH = args.candidates
        print(f"从命令行参数设置候选项数量: {c.CANDIDATES_PER_BATCH}")
    
    main(max_steps=args.max_steps, batch_size=args.batch_size)
