#!/bin/bash

# 自动进入 tmux，激活 conda，再运行你的联邦学习脚本

SESSION="myjob"
CONDA_INIT="$HOME/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="Hunyuan3D-Part"
WORKDIR="/home/yyz/Fedavg"
SCRIPT="run_online_learning.py"

# 如果 session 已经存在，先杀掉（可选）
tmux has-session -t $SESSION 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION already exists. Killing it..."
    tmux kill-session -t $SESSION
fi

echo "Starting tmux session '$SESSION'..."

tmux new-session -d -s $SESSION "
    source $CONDA_INIT && \
    conda activate $CONDA_ENV && \
    cd $WORKDIR && \
    python $SCRIPT
"

echo "Started! Use this command to view training logs:"
echo "tmux attach -t $SESSION"
