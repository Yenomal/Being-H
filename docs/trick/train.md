# 训练过程中的小妙招：

1. 在服务器上发现显存大、利用率低，是因为数据通过CPU处理，瓶颈变成了CPU，调整**NUM_WORKERS、PREFETCH_FACTOR**，让数据在CPU上并行处理，GPU利用率一下就上来了

2. beingh是余弦调度，所以必须设定一个比较好的总步数，推荐参数：MAX_STEPS=20000；SAVE_STEPS=2000，这个需要和batch对应，batch增大本身就会更收敛和稳定，所以可以缩小总步数

3. beingh通过一个batch的最大token来控制batch，推荐参数：MAX_NUM_TOKENS=8704；EXPECTED_NUM_TOKENS=8192；PREFER_BUFFER_BEFORE=4096；这一套是根据example的train来设置的