

**pytorch下载网址**

https://pytorch.org/get-started/locally/

老版本

https://pytorch.org/get-started/previous-versions/

**使用shell查看**

```shell
#显卡驱动信息，主要看CUDA支持的最高版本
nvidia-smi

#当前使用的CUDA的版本
nvcc -V

#查看安装了几个CUDA，当前使用哪个版本的CUDA
ll /usr/local/

#查看已安装的包的版本
conda list | grep cuda
conda list | grep torch

```

**使用py脚本查看**

vim version.py

```python
import torch
print(torch.__version__) # 查看torch版本
print(torch.cuda.is_available()) # 看安装好的torch和cuda能不能用，也就是看GPU能不能用
print(torch.version.cuda) # 输出一个 cuda 版本，注意：上述输出的 cuda 的版本并不一定是 Pytorch 在实际系统上运行时使用的 cuda 版本，而是编译该 Pytorch release 版本时使用的 cuda 版本，详见：https://blog.csdn.net/xiqi4145/article/details/110254093

import torch.utils
import torch.utils.cpp_extension
print(torch.utils.cpp_extension.CUDA_HOME) #输出 Pytorch 运行时使用的 cuda
```

**推算合适的pytorch和cuda版本**

**如何解决PyTorch版本和CUDA版本不匹配的关系 - 知乎 (zhihu.com)** 

https://zhuanlan.zhihu.com/p/633473214

核心步骤：

1. 根据GPU型号，去官网CUDA GPUs上去查询版本号，下图1中显示，RTX 3090的计算能力架构版本号是8.6，对应sm_86。其中8是主版本号，6是次版本号。
2. 仍然是上面的网页中，点链接进去，可查看到该GPU的架构。比如RTX 3090架构为Ampere
3. 根据架构，从下图2中查到CUDA版本范围，比如Ampere为CUDA 11.0-12.2
4. 项目一般会指定PyTorch版本，然后去PyTorch官网Start Locally | PyTorch找到PyTorch和CUDA的交集，选择CUDA最高的（运算更快）
5. 官方提供的一般是pip方式安装，如果慢，可尝试换源、代理等方式。
6. 除了pip安装方式，也可以whl方式下载离线安装包：

```
以Windows下为例。

假设在pytorch获得的pip安装命令为：
pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

如何获取whl离线安装包并安装？

下载地址：https://download.pytorch.org/whl/torch_stable.html，下载以下安装包：

torch-1.7.0+cu110-cp37-cp37m-win_amd64.whl
torchvision-0.8.1+cu110-cp37-cp37m-win_amd64.whl
torchaudio-0.7.0-cp37-none-win_amd64.whl

注意：cu110表示CUDA是11.0版本的，cp37表示python3.7，win表示windows版本，具体选择什么版本，可以参考上图中的“Run this Command”。

安装方法：进入离线安装包所在位置，然后“shift+鼠标右键”，然后选择“在此处打开powershell窗口”，最后输入“pip install torch-1.7.0+cu110-cp37-cp37m-win_amd64.whl”，即输入“pip install xxxx.whl”。

有可能会出现[winError]拒绝访问的错误提示，并且要求你添加“--user”，你可以这样输入：&quot;pip install xxxx.whl --user
```

**深入了解cuda、cudatoolkit以及多版本cuda共存时pytorch调用哪个**

进一步，你有必要深入了解一下cuda、cudatoolkit以及多版本cuda共存时pytorch调用哪个 cuda和cudatoolkit-CSDN博客

https://blog.csdn.net/xiqi4145/article/details/110254093

**安装需要的CUDA，多版本共存，并自由切换！**

安装需要的CUDA，多版本共存，并自由切换！ 【多版本cuda自由切换】在ubuntu上安装多个版本的CUDA，并且可以随时切换cuda-11.3//cuda-11.8//cuda-11.6//cuda-11.2_ubuntu切换cuda-CSDN博客 

https://blog.csdn.net/BetrayFree/article/details/134870198

注意：

1. 安装包类型要选择runfile，其它二者据说会有一些自动升级的行为，比较麻烦。
2. 实际安装过程中，我选择了驱动，但是没选择kernel objects、nvidia-fs
3. 可能会报nvidia的错误，看下面的处理

**nvidia报错的处理**

在安装过程中，会遇到报错,nvidia驱动需要卸载，参考： CUDA、驱动安装与踩坑记录 - 知乎 (zhihu.com) ubuntu升级NVIDIA驱动，遇到ERROR: An NVIDIA kernel module ‘nvidia-uvm‘ appears to already be loaded in your_error: an nvidia kernel module 'nvidia' appears to-CSDN博客

https://zhuanlan.zhihu.com/p/642632372

https://blog.csdn.net/hjxu2016/article/details/135128492



**CUDA下载**

1.访问 [NVIDIA CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads)。

选择以下选项：

- **Operating System**: Linux
- **Architecture**: x86_64
- **Distribution**: Linux
- **Version**: 选择 "Runfile (local)"，因为 `.run` 文件可以在没有 `sudo` 权限的情况下安装。

在服务器上，你可以使用 `wget` 命令直接下载：

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```

2. **解压并安装到本地用户目录**

1. **给 .run 文件赋予可执行权限**：

   ```bash
   chmod +x cuda_11.8.0_520.61.05_linux.run
   ```

2. **运行安装文件**（不使用 `sudo`）：

   ```bash
   ./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --override --installpath=$HOME/cuda-11.8
   ```

   - 这会将 CUDA Toolkit 安装到你用户目录的 `$HOME/cuda-11.8` 中。
   - 选项说明：
     - `--silent`：静默安装模式，不需要交互。
     - `--toolkit`：只安装 CUDA Toolkit，不安装驱动（因为没有 `sudo` 权限）。
     - `--override`：忽略权限问题。
     - `--installpath`：指定安装路径为用户目录下的 `cuda-11.8`。

3. **验证安装**：

   ```
   ls $HOME/cuda-11.8
   ```

   你应该能看到 `bin`, `lib64` 等文件夹。

**3. 配置环境变量**

要确保你的系统能够找到 CUDA 可执行文件和库，需要设置环境变量。

- **编辑 `.bashrc` 文件**：

  ```bash
  nano ~/.bashrc
  ```

- **在文件末尾添加以下内容**：

  ```bash
  export PATH=$HOME/cuda-11.8/bin:$PATH
  export LD_LIBRARY_PATH=$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH
  ```

- **使配置生效**：

  ```bash
  source ~/.bashrc
  ```

**4. 验证 CUDA 安装**

- **查看 CUDA 版本**：

  ```bash
  nvcc --version
  ```

  输出应类似于：

  ```
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on ....
  Cuda compilation tools, release 11.8, V11.8.89
  Build cuda_11.8.r11.8/compiler.31833905_0
  ```



改回11.7

```
export PATH=/home/wangchichu/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/home/wangchichu/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

 

**flash-attention安装**

官网下载https://github.com/Dao-AILab/flash-attention/releases对应版本的flash-attn

```bash
pip install flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
#验证是否安装成功
```

