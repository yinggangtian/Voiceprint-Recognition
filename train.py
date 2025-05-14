# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# merged test: 80 speaker, 5323 utterance
# batchisize 32*3 : train on triplet: 5s - > 3.1s/steps , softmax pre train: 3.1 s/steps

import logging
import argparse
from time import time
import numpy as np
import sys
import os
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import constants as c
from constants import *
import select_batch
from pre_process import data_catalog, preprocess_and_save
from models import *
from random_batch import stochastic_mini_batch
from triplet_loss import *
from utils import get_last_checkpoint, create_dir_and_delete_content
from test_model import eval_model

def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []

    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])

    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers
#-----------------------------------------------------------------
unique_speakers = 0
spk_index = None
#------------------------------------------------------------------------------------------

def main(libri_dir=c.DATASET_DIR, max_steps=None):
    PRE_TRAIN = c.PRE_TRAIN
    # 不要覆盖libri_dir，使用传入的默认值c.DATASET_DIR
    logging.info('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir, pattern='**/*.npy')

    if len(libri) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        logging.warning('Source directory: %s, Target directory: %s', c.WAV_DIR, c.DATASET_DIR)
        # 确保目标目录存在
        os.makedirs(c.DATASET_DIR, exist_ok=True)
        os.makedirs(os.path.join(c.DATASET_DIR, 'train-clean-100'), exist_ok=True)
        os.makedirs(os.path.join(c.DATASET_DIR, 'test-clean'), exist_ok=True)
        
        # 执行特征提取
        preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        
        # 重新加载数据集
        libri = data_catalog(libri_dir, pattern='**/*.npy')
        if len(libri) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            logging.warning('或者手动运行 python data_download_mini.py 下载和处理小型数据集')
            exit(1)

    global unique_speakers
    unique_speakers = np.sort(libri['speaker_id'].unique())
    print(unique_speakers)

    global spk_index
    spk_index = dict(zip(unique_speakers, range(len(unique_speakers))))
    print(spk_index)

    spk_utt_dict, unique_speakers = create_dict(libri['filename'].values, libri['speaker_id'].values,unique_speakers)


    select_batch.create_data_producer(unique_speakers, spk_utt_dict)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]
    train_batch_size = batch_size
    #batch_shape = [batch_size * num_frames] + list(b.shape[1:])  # A triplet has 3 parts.
    input_shape = (num_frames, b.shape[1], b.shape[2])

    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('input shape: {}'.format(input_shape))
    logging.info('x.shape : {}'.format(x.shape))
    orig_time = time()
    try:
        # 添加详细日志，帮助调试形状问题
        logging.info('构建模型时使用的输入形状: {}'.format(input_shape))
        
        # 确保输入形状合理
        if num_frames is None or num_frames <= 0:
            logging.warning('检测到无效的帧数: {}，使用默认值64'.format(num_frames))
            num_frames = 64
            input_shape = (num_frames, b.shape[1], b.shape[2])
            
        model = convolutional_model(input_shape=input_shape)
        logging.info(model.summary())
    except Exception as e:
        logging.error('构建模型时出错: {}'.format(e))
        logging.error('输入形状: {}'.format(input_shape))
        logging.error('请检查数据预处理和批处理的形状是否匹配')
        raise
    
    #----------------------------------------------------------------------------------------------
    gru_model = None
    if c.COMBINE_MODEL:
        if use_aamsoftmax_loss:
            gru_model = recurrent_model(input_shape=input_shape)
        
        elif use_sigmoid_cross_entropy_loss:
            gru_model = recurrent_model_sigmoid_cross_entropy(input_shape=input_shape, num_frames=num_frames, num_spks=len(unique_speakers))
        
        elif use_cross_entropy_loss:
            gru_model = recurrent_model_cross_entropy(input_shape=input_shape, num_frames=num_frames, num_spks=len(unique_speakers))
        
        elif use_triplet_loss:
            gru_model = recurrent_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
        
        elif user_center_loss:
            gru_model = recurrent_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
        
        elif use_coco_loss:
            gru_model = recurrent_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
        
        elif use_softmax_loss:
            gru_model = recurrent_model_softmax(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames, num_spks=len(unique_speakers))


            
        logging.info(gru_model.summary())
    grad_steps = 0

    if PRE_TRAIN:
        last_checkpoint = get_last_checkpoint(c.PRE_CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found pre-training checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            x = model.output
            x = Dense(len(unique_speakers), activation='softmax', name='softmax_layer')(x)
            pre_model = Model(model.input, x)
            pre_model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('Successfully loaded pre-training model')

    else:
        last_checkpoint = get_last_checkpoint(c.CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('[DONE]')
        if c.COMBINE_MODEL:
            last_checkpoint = get_last_checkpoint(c.GRU_CHECKPOINT_FOLDER)
            if last_checkpoint is not None:
                logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
                gru_model.load_weights(last_checkpoint)
                logging.info('[DONE]')
    
    #keras.losses.categorical_crossentropy
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #使用triplet loss 和 softmax loss　分别计算
    # model.compile(optimizer='adam', loss=deep_speaker_loss)
    # if c.COMBINE_MODEL:
    #     gru_model.compile(optimizer='adam', loss=deep_speaker_loss)

    model.compile(optimizer='adam', loss=deep_speaker_loss)

    if c.COMBINE_MODEL:
        if use_aamsoftmax_loss:
            gru_model.compile(optimizer='adam', loss=AAM_loss(len(unique_speakers)))
        
        elif use_sigmoid_cross_entropy_loss:
            gru_model.compile(optimizer='adam', loss=sigmoid_cross_entropy_loss(len(unique_speakers)))
        
        elif use_cross_entropy_loss:
            gru_model.compile(optimizer='adam', loss=cross_entropy_loss(len(unique_speakers)))

        elif use_softmax_loss:
            gru_model.compile(optimizer='adam', loss=softmax_loss(len(unique_speakers)))

        elif use_triplet_loss:
            gru_model.compile(optimizer='adam', loss=deep_speaker_loss)   
        
        elif user_center_loss:
            gru_model.compile(optimizer='adam', loss=center_loss(len(unique_speakers)))

        elif use_coco_loss:
            gru_model.compile(optimizer='adam', loss=coco_loss(len(unique_speakers)))


            
    print("model_build_time",time()-orig_time)
    logging.info('Starting training...')
    lasteer = 10
    eer = 1
    start_time = time()
    while True:
        orig_time = time()
        x, y = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
        y_true = [spk_index[one_id] for one_id in y]
        print("select_batch_time:", time() - orig_time)

        # Show progress information if max_steps is provided
        if max_steps is not None:
            progress_percent = (grad_steps / max_steps) * 100
            elapsed_time = time() - start_time
            estimated_total_time = elapsed_time / (grad_steps + 1) * max_steps
            estimated_remaining_time = estimated_total_time - elapsed_time
            
            logging.info('== Presenting step #{0}/{1} ({2:.1f}%) - Est. time remaining: {3:.2f}s'.format(
                grad_steps, max_steps, progress_percent, estimated_remaining_time))
        else:
            logging.info('== Presenting step #{0}'.format(grad_steps))
        orig_time = time()

        # Enable the CNN model training if needed (currently commented out)
        # loss = model.train_on_batch(x, y_true)
        loss = 0
        logging.info('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))
        
        # Train the GRU model if combined approach is enabled
        if c.COMBINE_MODEL:
            train_start = time()
            loss1 = gru_model.train_on_batch(x, y_true)
            train_time = time() - train_start
            logging.info('== Processed in {0:.2f}s by the gru-network, training loss = {1}.'.format(train_time, loss1))
            with open(c.GRU_CHECKPOINT_FOLDER + '/losses_gru.txt', "a") as f:
                f.write("{0},{1}\n".format(grad_steps, loss1))
        # record training loss
        with open(c.LOSS_LOG, "a") as f:
            f.write("{0},{1}\n".format(grad_steps, loss))
        if (grad_steps) % 10 == 0:
            fm1, tpr1, acc1, eer1, frr, far  = eval_model(model, train_batch_size, test_dir=c.DATASET_DIR, check_partial=True, gru_model=gru_model)
            logging.info('test training data EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer1, fm1, acc1))
            with open(c.CHECKPOINT_FOLDER + '/train_acc_eer.txt', "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer1, fm1, acc1))

        #fm, tpr, acc, eer, frr, far                
        if (grad_steps ) % c.TEST_PER_EPOCHS == 0 :
            fm, tpr, acc, eer, frr, far  = eval_model(model,train_batch_size, test_dir=c.TEST_DIR,gru_model=gru_model)
            logging.info('== Testing model after batch #{0}'.format(grad_steps))
            logging.info('EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer, fm, acc))
            with open(c.TEST_LOG, "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer, fm, acc))

        # checkpoints are really heavy so let's just keep the last one.
        if (grad_steps ) % c.SAVE_PER_EPOCHS == 0:
            create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.weights.h5'.format(c.CHECKPOINT_FOLDER, grad_steps, loss))
            if c.COMBINE_MODEL:
                gru_model.save_weights('{0}/grumodel_{1}_{2:.5f}.weights.h5'.format(c.GRU_CHECKPOINT_FOLDER, grad_steps, loss1))
            if eer < lasteer:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f), os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                lasteer = eer
                for file in files[:-4]:
                    logging.info("removing old model: {}".format(file))
                    os.remove(file)
                model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_model{0}_{1:.5f}.weights.h5'.format(grad_steps, eer))
                if c.COMBINE_MODEL:
                    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                          map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f),
                                              os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                                   key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                    lasteer = eer
                    for file in files[:-4]:
                        logging.info("removing old model: {}".format(file))
                        os.remove(file)
                    gru_model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_gru_model{0}_{1:.5f}.weights.h5'.format(grad_steps, eer))

        grad_steps += 1
        
        # Check if we've reached the maximum number of steps
        if max_steps is not None and grad_steps >= max_steps:
            logging.info(f'Reached maximum number of steps ({max_steps}). Training complete.')
            break



if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train the voiceprint recognition model.')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps')
    args = parser.parse_args()
    
    main(max_steps=args.max_steps)
