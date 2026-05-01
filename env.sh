# 写在前面：
# 1. 这个脚本用来从零配置python环境
# 2. 请保证你的系统环境安装好了conda+cuda12.1或cuda12.8
# 3. 请保证你运行这个脚本的路径是Being-H

# 创建并激活conda环境
conda create -n beingh python=3.10 -y
conda activate beingh

# 安装torch requirements
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ./Being-H05/requirements.txt

# 检测ABI
ABI=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null || echo "TRUE")

# 构建URL并安装
BASE_URL="https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.5/cu12/abi${ABI}/cp310"
WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"

# 安装flash attention
pip install "${BASE_URL}/${WHEEL_NAME}"



# 如果你是cuda12.8

# # 创建并激活conda环境
# conda create -n beingh python=3.10 -y
# conda activate beingh

# # 安装torch requirements
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# pip install -r ./Being-H05/requirements.txt

# # 检测ABI
# ABI=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null || echo "TRUE")

# # 构建URL并安装
# BASE_URL="https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.7/cu12/abi${ABI}/cp310"
# WHEEL_NAME="flash_attn-2.8.3+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"

# pip install "${BASE_URL}/${WHEEL_NAME}"