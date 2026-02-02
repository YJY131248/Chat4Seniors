conda create -n verl python=3.10
conda activate verl

# install verl and dependencies
sudo dnf install git -y
cd /devdata/yaojinyu/Chat4Seniors/verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

# update env
pip install tensordict
pip install torchdata
pip install msgspec
pip install vllm==0.8.5.post1 --user
pip install --upgrade "transformers>=4.51.0" --force-reinstall
pip install nvidia-ml-py==11.495.46 

# install model
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir /devdata/yaojinyu/Chat4Seniors/model/base_models/Qwen3-8B

# train
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NVML_BASED_CUDA_CHECK=0
bash /devdata/yaojinyu/Chat4Seniors/verl/personal/qwen3_8b_grpo.sh