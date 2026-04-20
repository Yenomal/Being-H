# beingh05 baseline

## 实验目的

- 跑通beingh05原始数据通路

- 记录beingh05外部数据流动情况（将beingh05视为黑箱，查看数据输入、config配置、数据输出情况）

## 实验过程

1. 利用bread数据测试train通路情况，设定epoch=1，确认raw数据到模型的预处理过程、ckpt的载入等外部过程

2. 利用libero Benchmark测试Inference通路情况，确定输出数据格式，以及模型输出到需要输出的后处理过程

## 下一步计划

## 相关脚本记录

train（）