数据集：LibriSpeech/voxceleb
模型：基于GRU

本工程使用步骤：
１．预处理

```
转换成wav格式
bash convert_flac_2_wav.sh
```
```
处理wav文件
python pre_process.py
需要同时处理训练集和测试集
```
2.配置相关参数
在constants.py中，配置数据集路径和训练时用的损失函数
```python
#使用LibriSpeech数据集
#Google drive
#soundRecong/librispeech/train——————>>/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100
#soundRecong/librispeech/test——————>>/Deep_Speaker_exp/audio/LibriSpeechTest/test-clean

DATASET_DIR = '/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100-npy/'
TEST_DIR = '/Deep_Speaker_exp/audio/LibriSpeechTest/test-clean-npy/'
WAV_DIR = '/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100/'
#使用voxceleb数据集
Google drive
#soundRecong/voxceleb-small-dataset/train————>>/Deep_Speaker_exp/audio/voxceleb/vox_train/
#soundRecong/voxceleb-small-dataset/test————>>/Deep_Speaker_exp/audio/voxceleb/vox_test/

# DATASET_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_train_npy/'
# TEST_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test_npy/'
# WAV_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test/'
#==============================================================

#训练集中说话人个数
NUM_SPEAKERS = 40

#根据要训练的loss来设置，只能有一个true
use_sigmoid_cross_entropy_loss = False
use_softmax_loss = False
use_coco_loss = False
use_aamsoftmax_loss = False
use_cross_entropy_loss = False
use_triplet_loss = False
user_center_loss = True

```

3.训练
```
python train.py
```


3.测试模型性能

```python
python eval_metrics.py
```

**使用softmax损失函数结果：**





**使用三元损失函数结果：**

模型在测试集上的结果：(全部数据集)
```
Found 0002620 files with 00040 different speakers.
f-measure = 0.18978102189762672, true positive rate = 0.9285714285714286, accuracy = 0.8414285714285709, equal error rate = 0.13265306122448978, frr=0.1071428571428571, far=0.15816326530612246
```

4.绘制曲线

训练过程中的损失曲线：
![](figs/loss.png)

在测试集上的准确率曲线：
![](figs/acc.png)

F值，EER，ACC放在一起：
![](figs/acc_eer.png)

