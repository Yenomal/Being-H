# beingh05_adapter
- 在baseline基本跑通的情况下，我们针对性地做一下调整

## 实验目的
- 调节Observation输入适应模型
- 修改action chunk到64（看看他的chunk是怎么设计的，基于当前state还是自回归生成，不能用自回归要一次生成）
- 使用RTC完成实时化，依赖chunk会导致误差

## 实验过程
1. Observation采用裁剪480+action chunk=64
```text
只需要修改参数FORCE_IMAGE_SIZE=480，ACTION_CHUNK_LENGTH=64，这个ACTION_CHUNK_LENGTH和输出action token不对齐，但是程序会帮助调整
```

2. resize到480
```text
这一步需要额外写一个脚本作为预处理存在，有以下几点原因单独写一个脚本：
1. resize和直接裁剪的效果其实需要做对比实验
2. resize可能影响模型运行时间
```

3. 

## 相关脚本记录