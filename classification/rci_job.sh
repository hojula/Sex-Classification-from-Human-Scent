#!/usr/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p &PARTITION # partition (queue)
#SBATCH --gres=gpu:&GPUS_PER_NODE
#SBATCH --exclude=g07
#SBATCH --cpus-per-task=&CPU_PER_TASK
#SBATCH --mem-per-cpu=&MEM_PER_CPU
#SBATCH --nodes=&NODES
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=M/F
#SBATCH --time=&TIME
#SBATCH -o &LOGSDIR/slurm.TEST_&BATCH_SIZE_%A.%a_%N.out # STDOUT
#SBATCH -e &LOGSDIR/slurm.TEST_&BATCH_SIZE_%A.%a_%N.err # STDERR

#set your working directory
cd "/home/moq/some/scent_release/identification/" || exit

#set your python path
export PYTHONPATH="/home/moq/some/scent/cvpr/identification/:${PYTHONPATH}"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

#set your singularity image name
SINGULARITY_IMAGE_NAME="scent"

# if env variable SLURM_JOB_ID is not defined, then define it
if [ -z "${SLURM_JOB_ID}" ]; then
  SLURM_JOB_ID="0000000"
fi

# if number of arguments is zero, echo "zero"
if [ $# -eq 0 ]; then
  echo "zero"
  #terminate
  exit 1
else
  NNODES=&NODES
  GPUS_PER_NODE=&GPUS_PER_NODE
  SCRIPT_PATH=&SCRIPT_PATH
  CONFIG_FILE="${1}"
  PROJECT_DIR="${2}"
  DATA_DIR="${3}"
  SCRATCH_DIR="${4}"
  SAVE_DIR="${5}"
  TXT_DIR="${6}"
  MASK_PATH="${7}"
  PLOT_DIR="${8}"

  echo "CONFIG_FILE: $CONFIG_FILE"
  echo "PROJECT_DIR: $PROJECT_DIR"
  echo "DATA_DIR: $DATA_DIR"
  echo "SCRATCH_DIR: $SCRATCH_DIR"
  echo "SAVE_DIR: $SAVE_DIR"
  echo "TXT_DIR: $TXT_DIR"
  echo "MASK_PATH: $MASK_PATH"
  echo "PLOT_DIR: $PLOT_DIR"
fi

# get node list and set master_node/master_address
declare -A node_global_rank
node_list=$(scontrol show hostnames "${SLURM_NODELIST}")
index=0
for node in ${node_list[@]}; do
    node_global_rank["${node}"]=${index}
    index=$((index+1))
done

master_node="$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)"

export NCCL_P2P_LEVEL=NVL
export SLURM_JOB_ID="${SLURM_JOB_ID}"

for node in ${node_list[@]}; do
    singularity exec --nv --cleanenv \
    -B"${SCRATCH_DIR}:${SCRATCH_DIR}" \
    -B"${PROJECT_DIR}:${PROJECT_DIR}" \
    -B"${DATA_DIR}:${DATA_DIR}" \
    -B"${SAVE_DIR}:${SAVE_DIR}" \
    -B"${TXT_DIR}:${TXT_DIR}" \
    -B"${MASK_PATH}:${MASK_PATH}" \
    -B"${GRID_DIR}:${GRID_DIR}" \
    -B"${PLOT_DIR}:${PLOT_DIR}" \
    "${PROJECT_DIR}/${SINGULARITY_IMAGE_NAME}.sif" \
    torchrun \
     --nnodes=${NNODES} \
     --nproc-per-node=${GPUS_PER_NODE} \
     --rdzv-id=${SLURM_JOB_ID} \
     --rdzv-backend=c10d \
     --rdzv-endpoint=${master_node} \
    ${SCRIPT_PATH} ${CONFIG_FILE} ${SLURM_JOB_ID} --clear_cache 2>&1 &
done
wait
