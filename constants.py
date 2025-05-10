#coding:utf-8
#==============================================================
#使用LibriSpeech数据集
DATASET_DIR = 'E:/mingde-AI/Deep_Speaker_exp3/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100-npy/'
TEST_DIR = 'E:/mingde-AI/Deep_Speaker_exp3//Deep_Speaker_exp/audio/LibriSpeechTest/test-clean-npy/'
WAV_DIR = 'E:/mingde-AI/Deep_Speaker_exp3//Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100/'

#使用voxceleb数据集
# DATASET_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_train_npy/'
# TEST_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test_npy/'
# WAV_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test/'
#==============================================================
BATCH_SIZE = 32        #must be even
TRIPLET_PER_BATCH = 3

SAVE_PER_EPOCHS = 200
TEST_PER_EPOCHS = 200
CANDIDATES_PER_BATCH = 640       # 18s per batch
TEST_NEGATIVE_No = 99


NUM_FRAMES = 160   # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
ALPHA = 0.2
HIST_TABLE_SIZE = 10

#要说话人个数
NUM_SPEAKERS =100


DATA_STACK_SIZE = 10

CHECKPOINT_FOLDER = 'checkpoints'
BEST_CHECKPOINT_FOLDER = 'best_checkpoint'
PRE_CHECKPOINT_FOLDER = 'pretraining_checkpoints'
GRU_CHECKPOINT_FOLDER = 'gru_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/acc_eer.txt'

PRE_TRAIN = False

COMBINE_MODEL = True

#根据要训练的loss来设置，只能有一个true
use_sigmoid_cross_entropy_loss = False
use_softmax_loss = True
use_coco_loss = False
use_aamsoftmax_loss = False
use_cross_entropy_loss = False
use_triplet_loss = False
user_center_loss = False
