## Dataset: LibriSpeech/VoxCeleb
## Model: GRU-based

### Project Usage Steps:

#### 1. Preprocessing
```bash
# Convert to wav format
bash convert_flac_2_wav.sh
```

```python
# Process wav files
python pre_process.py
# Need to process both training and test sets
```

#### 2. Configure Parameters
Configure dataset paths and loss functions for training in constants.py:

```python
# Using LibriSpeech dataset
# Google drive
# soundRecong/librispeech/train ——————>> /Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100
# soundRecong/librispeech/test ——————>> /Deep_Speaker_exp/audio/LibriSpeechTest/test-clean
DATASET_DIR = '/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100-npy/'
TEST_DIR = '/Deep_Speaker_exp/audio/LibriSpeechTest/test-clean-npy/'
WAV_DIR = '/Deep_Speaker_exp/audio/LibriSpeechSamples/train-clean-100/'

# Using VoxCeleb dataset
# Google drive
# soundRecong/voxceleb-small-dataset/train ————>> /Deep_Speaker_exp/audio/voxceleb/vox_train/
# soundRecong/voxceleb-small-dataset/test ————>> /Deep_Speaker_exp/audio/voxceleb/vox_test/
# DATASET_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_train_npy/'
# TEST_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test_npy/'
# WAV_DIR = '/Deep_Speaker_exp/audio/voxceleb/vox_test/'

#==============================================================
# Number of speakers in training set
NUM_SPEAKERS = 40

# Set according to the loss function to be trained, only one can be true
use_sigmoid_cross_entropy_loss = False
use_softmax_loss = False
use_coco_loss = False
use_aamsoftmax_loss = False
use_cross_entropy_loss = False
use_triplet_loss = False
user_center_loss = True
```

#### 3. Training
```bash
python train.py
```

#### 4. Test Model Performance
```python
python eval_metrics.py
```

**Results using softmax loss function:**

**Results using triplet loss function:**
Model results on test set: (full dataset)
```
Found 0002620 files with 00040 different speakers.
f-measure = 0.18978102189762672, true positive rate = 0.9285714285714286, accuracy = 0.8414285714285709, equal error rate = 0.13265306122448978, frr=0.1071428571428571, far=0.15816326530612246
```

#### 5. Plot Curves
Loss curve during training:
![](figs/loss.png)

Accuracy curve on test set:
![](figs/acc.png)

F-measure, EER, and ACC combined:
![](figs/acc_eer.png)
