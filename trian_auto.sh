# 写在前面
# 1. 这是一个自动化训练脚本，流程包括：数据预处理、模型训练
# 2. 请注意你的预处理过程，此处只有转lerobot
# 3. 请准备好你的数据和python环境
# 4. 需要手工配置好config，尤其是 Being-H05/configs/dataset_info.py 中的路径

conda activate hdf5_to_lerobot
python tool/hdf5-to-lerobot-converter/convert_hdf5_data_to_lerobot.py   --raw-dir ./datasets/raw/flower   --repo-id flower   --task "pick up the watering can and spray the flowers with water"   --mode video
bash Being-H05/scripts/train.sh

# # 数据推送与模型拉取，只作为参考，没啥用
# scp -r ./flower/raw root@nc12-lvrui-h20:/data/Being-H/datasets/raw/
# scp -r root@nc12-lvrui-h20:/data/Being-H/outputs/runs/flower/flower/0020000  ./