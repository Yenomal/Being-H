# beingh05 baseline

- BeingH05的相关信息参考[README.md](Being-H05/README.md)相关文件

## 实验目的

- 跑通beingh05原始数据通路

- 记录beingh05外部数据流动情况（将beingh05视为黑箱，查看数据输入、config配置、数据输出情况）

## 实验过程

1. data
```text
原始数据是hdf5，需要转化成lerobot格式，参考脚本来自https://github.com/LUhaotao/hdf5-to-lerobot-converter，针对灵巧手做了一些调整：
1. 关节状态 14 -> 任意维度（hdf5读取），命名修改成了joint_{id}
2. 相机维度 (3, 480, 640) -> 任意维度（hdf5读取）
这里需要注意使用video转换方式
```
PS. 输入过来的数据包括——末端位姿（7）+灵巧手（12），qpos是关节状态，action是控制指令，
- 待办：
- [x] 需要确认这里的hdf5具体数据内涵（velocity / effort是什么？接入不接入）
- [x] action和qpos是否同顺序同语义
- [x] 需要action是delta还是absolute —— absolute
- [x] 四元数的格式是wxyz还是xyzw —— xyzw
---

2. config
```text
在开始训练前需要申请与自己数据相关的config，此处的config配置如下：
1. Being-H05/configs/data_config.py：增加bread的config类 BreadDataConfig，DATA_CONFIG_MAP注册类
2. Being-H05/configs/dataset_info.py：DATASET_REGISTRY增加group名 bread_posttrain，DATASET_INFO增加lerobot数据路径
3. Being-H05/configs/posttrain/bread/bread.yaml：使用最基本的 bread.yaml 完成和 data_config.py、dataset_info.py 的对应即可

这组数据是四元数，需要使用四元数to轴角数据的转换脚本 Being-H05/configs/convert_quat_to_axis_angle.py
```
PS. 我们的灵巧手自由度 12 超过了预设的 6 自由度，只能使用extra的位置，这样不一定好用
- 待办：
- [x] 需要查看原文对于XHand的支持如何
- [] 确认dataset_info.py内的SftJSONLIterableDataset是什么数据集
---

3. train
```text
增加一个bread的训练脚本——Being-H05/scripts/train/train.py，需要修改这样一些参数：
1. 模型超参数（建议是完全不动超参数，尽量通过与数据相关的config和预处理解决问题）
2. 环境参数（主要的比如wandb、GPU）
3. InternVL3、Qwen3模型路径
4. Output路径
5. Datasets、Ckpt路径
```
PS. 通过token控制了batch量，模型的推理延迟大概150ms，拓宽图像可能会增大延迟，可以跑一个resize试试

4. inference
```text
在真机上部署C/S结构，步骤为：
1. 开启ROS
2. 开启server
3. 开启client
```
这一部分有点乱后面整理一个**封装好的client**

## 相关脚本记录

请注意所有的脚本需要在Being-H/下运行

- hdf5_to_lerobot：
在./tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py内设定输出路径

python tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py \
  --raw-dir ./datasets/raw/test_EE \
  --repo-id test_EE \
  --task "pick the bread and place it in the machine" \
  --mode video

python tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py   --raw-dir ./datasets/raw/test_EE   --repo-id test_EE   --task "pick the bread and place it in the machine"   --mode video

- quat四元数 to 轴角：

python Being-H05/configs/convert_quat_to_axis_angle.py \
  --dataset-root ./datasets/lerobot/test_EE \
  --quat-order xyzw

python Being-H05/configs/convert_quat_to_axis_angle.py   --dataset-root ./datasets/lerobot/test_EE   --quat-order xyzw

- 拉取qwen3、internvl3

hf download OpenGVLab/InternVL3_5-2B --local-dir ./ckpt/model/InternVL3_5-2B

hf download Qwen/Qwen3-0.6B --local-dir ./ckpt/model/Qwen3-0.6B

- train：

bash Being-H05/scripts/train/train.sh

## 注意事项

1. 需要pip install av -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 项目的cuda版本是13.0，但是我们的服务器最高只支持12.8

3. 接续上面2，我们重装torch后重装flash-attn，发现报错限制flash-attn二进制文件错误，这是因为flash-attn重装的时候缓存信息还在，so依然指向cuda13.0，需要先清理缓存信息

4. 记得配环境的时候增加镜像源，或者增加到本机的代理，最好增加镜像源：pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## 使用流程

此处记录如何使用自己的数据集进行Train、Inference过程，注意使用需要**保证自己在Being-H路径下**：

---
- Train，假设数据集为hdf5

1. 数据预处理：转hdf5数据到lerobot格式，根据slots格式调整数据维度以对齐slots

2. config设置：配置[data_config.py](Being-H05/configs/data_config.py)、[dataset_info.py](Being-H05/configs/dataset_info.py)，并增加一个yaml文件进行配置，参考[config_template.yaml](Being-H05/configs/posttrain/config_template.yaml)配置

3. 调整train bash脚本内容，主要是GPU相关信息

**针对不同的数据集，最关键的就是输入格式对齐、数据归一化（体现在config上）、任务相关设定（体现在bash脚本上）**

---
- Inference

1. 保证数据进行相同的预处理

2. 输出和底层输出之间的Adapter设计

--- 
- python环境

- 配置

1. 安装torch（同时安装配套的cuda，建议装12.1，比较稳定）
    1. cuda12.1：pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1（至少需要torch >= 2.5，因为flex_attention需要，装好了运行的时候可能也找不到，需要unset掉一个环境变量——unset LD_LIBRARY_PATH）
    2. cuda12.8：pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

2. 安装requirements.txt（需要将av放进requirements.txt，否则单独安装av）：pip install -r ./Being-H05/requirements.txt

3. 安装flash-attn：pip install flash-attn --no-build-isolation（配置flash attn需要torch+对应的cuda），flash-attn实时编译应该是有点问题，后面看看，可以问问ai有社区的wheel版本的，具体而言使用了这样的安装命令：

    1. 查看ABI情况
    ``` text
    python - <<'PY'
    import torch
    print(torch._C._GLIBCXX_USE_CXX11_ABI)
    PY
    ```
    2. 安装 flash-attn
    ``text
    ABI=true：pip install "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.5/cu12/abiTRUE/cp310/flash_attn-2.8.3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    ABI=false：pip install "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.5/cu12/abiFALSE/cp310/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    ```

## TODO

- [x] 为了更好地调节config，后续会增加一个config.yaml template方便填写，而在data_config和dataset_info内的config偏注册性质，不进行template

- [] 对齐config和data，看看有没有需要对data进行处理的部分

- [x] 对齐输出和底层控制

- [x] 整理train bash脚本和config适配

## 下一步计划

当前训练了三个实验——bread、bread_0，其中bread的batch小、步数少没开MPG，整体收敛比较好，bread_0做了MPG的消融实验但是在10k步左右收敛不了，是因为学习率使用余弦调度，而我们总步数设定了100k导致一直学习率太高，下一步我们将会在稳定的post-train上先做一点工作

- [] 参数调节 —— 小学习率、少总步数

- [] 看看减小总步数后怎么样，可以考虑增加一些别的归一化的trick，我们当前的state-action归一化还基本没做

- [] 尝试引入 RTC