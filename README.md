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



以下是我在代码中发现的一些潜在问题和改进建议：

1. 硬编码路径

许多文件中存在硬编码路径，例如：

constants.py 中的 DATASET_DIR, TEST_DIR, WAV_DIR。
pre_process_voxceleb.py 和 kaldi_form_preprocess.py 中的路径。
建议： 将路径配置集中到一个配置文件或环境变量中，避免硬编码路径。

2. 重复代码

多个文件中存在重复的功能实现，例如：

find_files 函数在 pre_process_voxceleb.py 和 kaldi_form_preprocess.py 中重复。
VAD 函数在多个文件中重复。
建议： 将这些通用函数提取到一个公共模块（如 utils.py）中，供其他模块调用。

3. 异常处理不足

许多函数缺乏异常处理，例如：

read_audio 和 extract_features 函数中未处理文件读取失败或音频格式错误的情况。
preprocess_and_save 中未处理多进程任务可能的失败。
建议： 为文件操作、音频处理等关键步骤添加异常处理，确保程序在异常情况下不会崩溃。

4. 多进程问题

在 pre_process.py 和 pre_process_voxceleb.py 中使用了 multiprocessing.Pool，但未设置 Pool 的上下文管理器，可能导致资源泄漏。

建议： 使用 with Pool(...) as p: 的形式管理进程池。

5. 日志记录不足

许多文件中使用了 print 语句记录信息，例如：

pre_process.py 和 pre_process_voxceleb.py 中的 print。
kaldi_form_preprocess.py 中的 print。
建议： 统一使用 logging 模块记录日志，便于调试和维护。

6. 硬编码超参数

许多超参数（如 BATCH_SIZE, NUM_FRAMES, SAMPLE_RATE 等）直接硬编码在代码中。

建议： 将这些超参数集中到配置文件（如 constants.py）中，便于统一管理和调整。

7. 未使用的代码

一些代码片段未被使用，例如：

silence_detector.py 中的 if __name__ == '__main__': 部分。
kaldi_form_preprocess.py 中的 test 函数。
建议： 清理未使用的代码，保持代码整洁。

8. 依赖版本问题

require.txt 中的依赖版本较旧，例如：

tensorflow==1.12.0 和 tensorflow-gpu==1.4.1 已经过时，可能与现代硬件或软件环境不兼容。
建议： 升级依赖库到最新版本，并测试兼容性。

9. 图像保存路径问题

utils.py 中的 plot_loss 和 plot_acc 函数将图像保存到 figs/ 目录，但未检查该目录是否存在。

建议： 在保存图像前检查目录是否存在，或使用 os.makedirs 创建目录。

10. 潜在的性能问题

pre_process.py 和 pre_process_voxceleb.py 中对大文件的逐行处理可能导致性能瓶颈。
train.py 中的模型保存逻辑可能会频繁删除和创建文件，影响性能。
建议： 优化文件处理逻辑，减少不必要的 I/O 操作。

11. 文档和注释不足

许多函数缺乏详细的文档字符串，代码的可读性较差。

建议： 为每个函数添加清晰的文档字符串，说明其功能、参数和返回值。

12. 测试覆盖不足

目前代码中缺乏单元测试或集成测试，可能导致代码在修改后出现不可预见的问题。

建议： 为关键模块（如 pre_process.py, train.py）编写测试用例，确保代码的正确性。

如果需要，我可以帮助修复这些问题或提供具体的代码实现。
