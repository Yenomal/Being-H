# beingh05 baseline

## 实验目的

- 跑通beingh05原始数据通路

- 记录beingh05外部数据流动情况（将beingh05视为黑箱，查看数据输入、config配置、数据输出情况）

## 实验过程

1. data
```text
原始数据是hdf5，需要转化成lerobot格式，参考脚本来自https://github.com/LUhaotao/hdf5-to-lerobot-converter，针对灵巧手做了一些调整（偏保守地进行固定设置）：
1. 关节状态 14 -> 任意维度（hdf5读取），命名修改成了joint_{id}
2. 相机维度 (3, 480, 640) -> 任意维度（hdf5读取）
```
PS. 输入过来的数据包括——末端位姿（7）+灵巧手（12），qpos是关节状态，action是控制指令，
- 待办：
- [] 需要确认这里的hdf5具体数据内涵（velocity / effort是什么？接入不接入）
- [] action和qpos是否同顺序同语义
- [x] 需要action是delta还是absolute
---
2. config
```text
在开始训练前需要申请与自己数据相关的config，此处的config配置如下：
1. Being-H05/configs/data_config.py：增加bread的config类 BreadDataConfig，DATA_CONFIG_MAP注册类
2. Being-H05/configs/dataset_info.py：DATASET_REGISTRY增加group名 bread_posttrain，DATASET_INFO增加lerobot数据路径
3. Being-H05/configs/posttrain/bread/bread.yaml：使用最基本的 bread.yaml 完成和 data_config.py、dataset_info.py 的对应即可
```
PS. 我们的灵巧手自由度 12 超过了预设的 6 自由度，只能使用extra的位置，这样不一定好用
- 待办：
- [] 需要查看原文对于XHand的支持如何
- [] 确认dataset_info.py内的SftJSONLIterableDataset是什么数据集
---
3. train
```text
增加一个bread的训练脚本——Being-H05/scripts/train/train_bread_test.py
```
## 下一步计划
1. 为了更好地调节config，后续会增加一个config.yaml template方便填写，而在data_config和dataset_info内的config偏注册性质，不进行template

## 相关脚本记录

请注意所有的脚本需要在Being-H/下运行

- hdf5_to_lerobot:
在./tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py内设定输出路径
uv run ./tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py \
  --raw-dir ./datasets/raw/test_EE \
  --repo-id test_EE \
  --task "pick the bread and place it in the machine" \
  --mode video

uv run ./tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py   --raw-dir ./datasets/raw/test_EE   --repo-id test_EE   --task "pick the bread and place it in the machine"   --mode video

- train：
