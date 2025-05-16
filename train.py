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
from time import time
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
    使用XLA JIT编译和其他优化策略优化模型
    """
    # 配置TensorFlow使用所有可用的CPU进行训练
    print("配置模型以最大化CPU利用率...")
    
    # 确保使用XLA优化
    tf.config.optimizer.set_jit(True)  # 启用XLA JIT
    
    # 设置线程和优化配置
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
    
    # 使用优化的Adam优化器
    optimizer = Adam(learning_rate=0.001, epsilon=1e-7, beta_1=0.9, beta_2=0.999)
    
    # 尝试使用混合精度训练
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
    if tf_version >= (2, 4) and hasattr(tf.keras, 'mixed_precision'):
        try:
            print("启用混合精度训练以提高性能...")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("混合精度训练已启用")
        except Exception as e:
            print(f"混合精度训练未启用: {e}")
    
    # 使用JIT编译优化模型编译过程
    if loss_fn is None:
        loss_fn = deep_speaker_loss
    
    # 使用tf.function替代jit_scope，这在Eager模式下是兼容的
    # 避免使用 tf.xla.experimental.jit_scope() 因为它在Eager模式下不兼容
    model.compile(
        optimizer=optimizer, 
        loss=loss_fn,
        # 在Keras编译中直接启用jit_compile，而不使用jit_scope
        jit_compile=True
    )
    
    return model

# 创建优化的训练步骤函数
def create_train_step(model):
    """创建XLA编译的训练步骤函数"""
    @tf.function(experimental_compile=True)
    def train_step(x, y):
        """使用XLA编译的训练步骤函数 - 提高CPU利用率"""
        return model.train_on_batch(x, y)
    return train_step

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
    
    # 设置全局批次大小
    c.BATCH_SIZE = batch_size
    
    # 计算总批次大小
    batch_size_total = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    print(f"使用批次大小: {c.BATCH_SIZE}, 总批次大小: {batch_size_total}")

    # Load or preprocess data
    logging.info(f"Looking for fbank features in {libri_dir}")
    libri = data_catalog(libri_dir, pattern='**/*.npy')
    if libri is None or len(libri) == 0:
        logging.warning("No npy files found, running preprocess...")
        # create dataset dirs
        os.makedirs(c.DATASET_DIR, exist_ok=True)
        os.makedirs(os.path.join(c.DATASET_DIR, 'train-clean-100'), exist_ok=True)
        os.makedirs(os.path.join(c.DATASET_DIR, 'test-clean'), exist_ok=True)
        preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        libri = data_catalog(libri_dir, pattern='**/*.npy')
        if libri is None or len(libri) == 0:
            logging.error("Preprocess failed, no data found. Exiting.")
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

    # Build sample batch to infer shapes
    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=speakers)
    x0, y0 = batch.to_inputs()
    num_frames = x0[0].shape[0]
    input_shape = (num_frames, x0[0].shape[1], x0[0].shape[2])
    logging.info(f"Input shape: {input_shape}, total batch size: {batch_size_total}")

    # Build models
    model = convolutional_model(input_shape=input_shape)
    # 使用XLA和其他优化技术优化CNN模型
    model = optimize_model(model, deep_speaker_loss)
    model.summary(print_fn=lambda s: logging.info(s))

    gru_model = None
    if c.COMBINE_MODEL:
        if c.use_softmax_loss:
            gru_model = recurrent_model_softmax(
                input_shape=input_shape,
                batch_size=batch_size_total,
                num_frames=num_frames,
                num_spks=len(speakers)
            )
            # 使用Softmax损失优化GRU模型
            gru_model = optimize_model(gru_model, softmax_loss(len(speakers)))
        else:
            gru_model = recurrent_model(
                input_shape=input_shape,
                batch_size=batch_size_total,
                num_frames=num_frames
            )
            # 使用深度说话人损失优化GRU模型
            gru_model = optimize_model(gru_model, deep_speaker_loss)
        gru_model.summary(print_fn=lambda s: logging.info(s))

    # Load checkpoints
    if c.PRE_TRAIN:
        cp = get_last_checkpoint(c.PRE_CHECKPOINT_FOLDER)
        if cp:
            logging.info(f"Loading pretrain checkpoint: {cp}")
            x = Dense(len(speakers), activation='softmax')(model.output)
            pre_model = Model(model.input, x)
            pre_model.load_weights(cp)
    else:
        cp = get_last_checkpoint(c.CHECKPOINT_FOLDER)
        if cp:
            model.load_weights(cp)
            logging.info(f"Loaded model checkpoint: {cp}")
        if c.COMBINE_MODEL:
            cp2 = get_last_checkpoint(c.GRU_CHECKPOINT_FOLDER)
            if cp2:
                gru_model.load_weights(cp2)
                logging.info(f"Loaded GRU checkpoint: {cp2}")

    # 创建XLA编译的训练步骤函数
    cnn_train_step = create_train_step(model)
    gru_train_step = create_train_step(gru_model) if c.COMBINE_MODEL else None

    # Training loop
    grad_steps = 0
    start = time()
    
    while True:
        try:
            # Select best batch - 优化选择批次过程
            t0 = time()
            x_batch, y_ids = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
            y_true = np.array([spk_index[i] for i in y_ids])
            batch_select_time = time()-t0
            
            # 资源使用监控
            resource_status = get_resource_usage()
            
            # Progress
            if max_steps:
                pct = grad_steps / max_steps * 100
                elapsed = time() - start
                rem = elapsed / (grad_steps+1) * (max_steps - grad_steps)
                logging.info(f"Step {grad_steps}/{max_steps} ({pct:.1f}%), rem {rem:.1f}s, batch_time={batch_select_time:.2f}s | {resource_status}")
            else:
                logging.info(f"Step {grad_steps}, batch_time={batch_select_time:.2f}s | {resource_status}")

            # 使用CNN训练步骤
            loss = 0
            if not c.COMBINE_MODEL or (c.COMBINE_MODEL and c.model_cnn_train):
                # 使用XLA编译后的训练步骤提高性能
                loss = cnn_train_step(x_batch, y_true).numpy()
                logging.info(f"CNN loss {loss:.4f}")

            # GRU train
            if c.COMBINE_MODEL:
                try:
                    # 使用XLA编译后的GRU训练步骤
                    loss1 = gru_train_step(x_batch, y_true).numpy()
                    logging.info(f"GRU loss {loss1:.4f}")
                    with open(os.path.join(c.GRU_CHECKPOINT_FOLDER, 'losses_gru.txt'), 'a') as f:
                        f.write(f"{grad_steps},{loss1}\n")
                except Exception as ex:
                    logging.error(f"GRU train error: {ex}")

            # Log CNN loss
            with open(c.LOSS_LOG, 'a') as f:
                f.write(f"{grad_steps},{loss}\n")

            # Periodic evaluation
            if grad_steps % 10 == 0:
                # Use recursive glob for test dir
                try:
                    fm, tpr, acc, eer, *_ = eval_model(
                        model,
                        batch_size_total,
                        test_dir=c.TEST_DIR,
                        pattern='**/*.npy',
                        gru_model=gru_model
                    )
                    logging.info(f"Eval EER {eer:.3f}, Fm {fm:.3f}, Acc {acc:.3f}")
                    with open(c.TEST_LOG, 'a') as f:
                        f.write(f"{grad_steps},{eer:.3f},{fm:.3f},{acc:.3f}\n")
                except Exception as eval_ex:
                    logging.error(f"Eval error: {eval_ex}")

            # Save checkpoints
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
            return

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description='训练声纹识别模型 (CPU优化版)')
    parser.add_argument('--max_steps', type=int, default=None, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=None, help='训练批次大小，越大越能充分利用CPU')
    args = parser.parse_args()
    main(max_steps=args.max_steps, batch_size=args.batch_size)
