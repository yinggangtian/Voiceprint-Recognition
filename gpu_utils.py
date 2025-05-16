#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import constants as c
import logging

def configure_gpu():
    """
    根据 constants.py 中的设置配置 GPU 环境
    - USE_GPU: 是否使用 GPU
    - GPU_MEMORY_LIMIT: GPU 内存限制
    - MIXED_PRECISION: 是否使用混合精度训练
    
    Returns:
        visible_devices: 可见的 GPU 设备列表
    """
    # 检查是否有可用的 GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if not physical_devices:
        logging.warning("没有找到可用的 GPU 设备")
        return []
    
    if not c.USE_GPU:
        logging.info("根据配置禁用 GPU")
        # 如果不使用 GPU，将所有 GPU 设为不可见
        tf.config.set_visible_devices([], 'GPU')
        return []
    
    # 设置 GPU 内存使用
    for device in physical_devices:
        try:
            if c.GPU_MEMORY_LIMIT:
                # 限制 GPU 内存使用
                tf.config.experimental.set_virtual_device_configuration(
                    device,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=c.GPU_MEMORY_LIMIT)]
                )
                logging.info(f"GPU {device.name} 内存限制设为 {c.GPU_MEMORY_LIMIT}MB")
            else:
                # 允许 GPU 内存按需增长
                tf.config.experimental.set_memory_growth(device, True)
                logging.info(f"GPU {device.name} 内存设为按需增长模式")
        except RuntimeError as e:
            logging.error(f"GPU 配置错误: {str(e)}")
    
    # 启用混合精度
    if c.MIXED_PRECISION:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("已启用混合精度训练 (float16)")
        except Exception as e:
            logging.error(f"无法启用混合精度: {str(e)}")
    
    # 设置 XLA 优化
    tf.config.optimizer.set_jit(True)
    
    # 返回可见的 GPU 设备列表
    return physical_devices

def print_gpu_info():
    """打印 GPU 信息"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            logging.info(f"使用 GPU: {gpu.name}")
            # 获取 GPU 详细信息
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                logging.info(f"GPU 详细信息: {gpu_details}")
            except:
                pass  # 如果获取详细信息失败，就跳过
    else:
        logging.info("未检测到 GPU，将使用 CPU 训练")
    
    logging.info(f"TensorFlow 版本: {tf.__version__}")
    logging.info(f"启用混合精度: {c.MIXED_PRECISION}")
