#coding:utf-8
#==============================================================
#使用LibriSpeech数据集
DATASET_DIR = './melspec_small/'
TEST_DIR = './melspec_small/test-clean/'
WAV_DIR = './audio/LibriSpeech/'

#使用voxceleb数据集
# DATASET_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_train_npy/'
# TEST_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test_npy/'
# WAV_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test/'
#==============================================================
# 原本可能只有 NUM_FRAMES，这里我们拆分成两个
NUM_MELS   = 64      # Mel 频带数
NUM_FRAMES = None    # 时间帧数：None 表示可变长度；也可改成整数，表示固定帧长
CHANNELS   = 1       # 单通道
BATCH_SIZE = 32      # 例子值，保持你原来的设置
TRIPLET_PER_BATCH = 3
CANDIDATES_PER_BATCH = 96  # 每批次的候选项数量，通常是 BATCH_SIZE 的 3 倍

SAMPLE_RATE = 16000  # Standard 16kHz sampling rate for speech processing
TRUNCATE_SOUND_SECONDS = (0.2, 1.2)  # Also add this if not present
ALPHA = 0.2  # Margin for triplet loss, used in FaceNet, typically between 0.2-0.3

# ─── 输入处理相关选项 ───
# PAD_MODE: 指定如何处理可变长度输入
# 'none': 保持原来长度，完全支持可变长度输入
# 'zero': 使用零填充将短序列填充到指定长度
# 'repeat': 通过重复最后一帧来填充到指定长度
# 'mirror': 通过镜像序列来填充到指定长度
PAD_MODE = 'repeat'  

# ─── 测试相关参数 ───
TEST_NEGATIVE_No = 50  # 测试时使用的负样本数量

# 错误恢复设置
OOM_RECOVERY_FACTOR = 0.6  # OOM错误后批次大小减少的比例 (0.6 = 原来的60%)
MIN_BATCH_SIZE = 4        # 最小可接受的批次大小
MAX_RETRIES = 5           # 最大重试次数

# 内存使用优化
DATA_STACK_SIZE = 10     # 预加载批次的数量
HIST_TABLE_SIZE = 10     # 历史表大小

#要说话人个数
NUM_SPEAKERS =100


DATA_STACK_SIZE = 10

CHECKPOINT_FOLDER = 'checkpoints'
BEST_CHECKPOINT_FOLDER = 'best_checkpoint'
PRE_CHECKPOINT_FOLDER = 'pretraining_checkpoints'
GRU_CHECKPOINT_FOLDER = 'gru_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/acc_eer.txt'

PRE_TRAIN = True

COMBINE_MODEL = True
CNN_MODEL_TRAIN = False
SAVE_PER_EPOCHS = 50   # 每多少轮保存一次检查点

# GPU 训练设置
USE_GPU = False           # 是否使用 GPU 训练
GPU_MEMORY_LIMIT = None  # GPU 内存限制，None 表示使用全部可用内存
MIXED_PRECISION = True   # 是否使用混合精度训练 (可提高训练速度)

#根据要训练的loss来设置，只能有一个true
use_sigmoid_cross_entropy_loss = False
use_softmax_loss = True
use_coco_loss = False
use_aamsoftmax_loss = False
use_cross_entropy_loss = False
use_triplet_loss = False
user_center_loss = False
