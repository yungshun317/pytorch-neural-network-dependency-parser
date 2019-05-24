# PyTorch Neural Network Dependency Parser

[![Made with Python](https://img.shields.io/badge/Made_with-Python-blue.svg)](https://img.shields.io/badge/Made_with-Python-blue.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

### Adam Optimizer
The **Adam (adaptive moment estimation) optimization algorithm** is an extension to stochastic gradient descent that has recently seen broader adopotion for deep learning applications in computer vision and natural language processing.

**Stochastic gradient descent** maintains a single **learning rate** (termed alpha) for all weight updates and the learning rate does not change during training. A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.

Adam combines the advantages of AdaGrad and RMSProp:
* **Adaptive Gradient Algorithm (AdaGrad):** maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
* **Root Mean Square Propagation (RMSProp):** also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).

Instead of adapting the parameter learning rates based on the average first moment (the mean) as in RMSProp, Adam also makes use of the average of the second moments of the gradients (the uncentered variance).

### Dropout
Simply put, **dropout** refers to ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random. By "ignoring", I mean these units are not considered during a particular forward or backward pass.

Why do we need dropout? The reason is to prevent **over-fitting**. Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.

### Neural Transition-Based Dependency Parsing

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between head words, and words which modify those heads. Your implementation will be a transition-based parser, which incrementally builds up a parse one step at a time. At every step it maintains a partial parse, which is represented as follows:
* A stack of words that are currently being processed.
* A buffer of words yet to be processed.
* A list of dependencies predicted by the parser.
Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words of the sentence in order. At each step, the parser applies a transition to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied:
* SHIFT: removes the first word from the buffer and pushes it onto the stack.
* LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack.
* RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack.


```sh
$ python run.py
...
Average Train Loss: 0.1461492809446859
Evaluating on dev set
1445850it [00:00, 26595158.57it/s]
- dev UAS: 76.93

================================================================================
TESTING
================================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set
2919736it [00:00, 27853743.40it/s]
- test UAS: 76.90
Done!
```

## CUDA, cuDNN, and GPU support

Setting up the environment to train deep neural network models with GPU is a tedious task. The following is an instruction to install TensorFlow and PyTorch with CUDA, cuDNN, and GPU support on Windows 10.

First setup CUDA and cuDNN.
1. Download and install [Visual Studio Express]( https://visualstudio.microsoft.com/vs/express/). According to the [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), to use CUDA you need to have supported Visual Studio installed, which is Visual Studio 2019 for me, but I suspect this step is not really necessary.
2. Download and install [the NVIDIA CUDA Toolkit](http://developer.nvidia.com/cuda-downloads). At this time, there is no patch released for CUDA 10.1.
3. Download and install [cuDNN]( https://developer.nvidia.com/cudnn). I use cuDNN v7.6.4 for CUDA 10.1. It's a bit strange that we have to copy the files manually not automatically.
    * Copy `cuda\bin\cudnn64_7.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\`.
    * Copy `cuda\include\cudnn.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include\`.
    * Copy `cuda\lib\x64\cudnn.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\`.
4. Check CUDA environment variables are set in Windows. Go to **Control Panel > System and Security > System > Advanced System Settings > Environment Variables**. Within **system variables**, click on **path** and choose the button **edit**. I found the two paths to CUDA had been added already.

Then we can start to install our deep learning packages. Note that I already have Anaconda 4.7.12 in place. My Python version is 3.6.9; `pip` version is 19.2.3. 
1. Install TensorFlow via Python `pip` in Anaconda Prompt.
```sh
$ pip install –upgrade tensorflow-gpu
```

Test.
```sh
$ python
>>> Import tensorflow as tf
```

If there is an error about `numpy`'s version, run
```sh
$ conda update --all
```
And then try the test again.

To test CUDA support for your TensorFlow installation, run
```sh
>>> tf.test.is_built_with_cuda()
True
```

Finally, to confirm that the GPU is available to TensorFlow, test with
```sh
>>> tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
2019-10-10 19:04:35.350712: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-10-10 19:04:35.723285: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1392] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.645
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2019-10-10 19:04:35.736445: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1471] Adding visible gpu devices: 0
2019-10-10 19:08:45.267607: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-10-10 19:08:45.278437: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:958]      0
2019-10-10 19:08:45.283925: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:971] 0:   N
2019-10-10 19:08:45.292952: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1084] Created TensorFlow device (/device:GPU:0 with 6399 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
True
```

2. Install Keras via Python `pip` in Anaconda Prompt.
```sh
$ pip install keras
```
To check which version of Keras is installed, run
```sh
$ python
>>> import keras 
Using TensorFlow backend.
>>> keras.__version__
'2.2.0'
```

3. Install PyTorch via Python `pip` in Anaconda Prompt.
```sh
$ pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

To check if PyTorch is using the GPU, run
```
$ python
>>> import torch
>>> torch.cuda.current_device()
0
>>> torch.cuda.device(0)
<torch.cuda.device object at 0x0000023DCAE9E588>
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name(0)
'GeForce GTX 1070'
>>> torch.cuda.is_available()
True
```

I definitely don't recommend using Windows 10 as your development environment. However, right now I only have a MSI GT72VR gaming laptop with NVIDIA GeForce GTX 1070 GPU as the only machine I can use.

## Todos
 - Learn NLP from Stanford's [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) and [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) materials.
 - Build a question answering system for competitions.

## License
[PyTorch Neural Network Dependency Parser](https://github.com/yungshun317/pytorch-neural-network-dependency-parser) is released under the [MIT License](https://opensource.org/licenses/MIT) by [yungshun317](https://github.com/yungshun317).