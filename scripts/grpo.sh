#PBS -q rt_HF
#PBS -l select=1:ncpus=96:ngpus=8:mpiprocs=8
#PBS -N grpo
#PBS -l walltime=45:00:00
#PBS -P gcc50533
#PBS -j oe
source /etc/profile.d/modules.sh
source ~/.bash_profile

set -e
cd $PBS_O_WORKDIR

module load cuda/12.4
source .venv_r1/bin/activate
date=`date '+%Y-%m-%d-%H-%M-%S'`
EXP_NAME="grpo"

PBS_NUMID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)

mkdir -p $PBS_O_WORKDIR/outputs
LOGFILE="$PBS_O_WORKDIR/outputs/${EXP_NAME}_$PBS_NUMID.log"
exec > "$LOGFILE" 2>&1


export MASTER_PORT=$((10000 + ($PBS_NUMID % 50000)))
export MASTER_ADDR=$(
  ip a show dev bond0 \
  | grep 'inet ' \
  | head -n 1 \
  | cut -d " " -f 6 \
  | cut -d "/" -f 1
)

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

NUM_GPU_PER_NODE=8
NUM_NODES=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${PBS_NUMID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "開始時間: $start_time"

mkdir -p /tmp/triton_cache
mkdir -p /tmp/hf_datasets_cache
#export TMPDIR="/tmp"
#export TRITON_CACHE_DIR="/tmp/triton_cache"
#export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --wandb_entity kawamuramasaki \
    --wandb_project open-r1 \
    --run_name DeepSeek-R1-Distill-Qwen-1.5B-GRPO-4



end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "終了時間: $end_time"
