#! /bin/bash

#SBATCH --job-name=evaluate          # 作业名称
#SBATCH --ntasks-per-node=1              # 每个节点上运行的任务数
#SBATCH --cpus-per-task=1                # 每个任务使用的CPU数
#SBATCH --gres=gpu:1                   # 分配2个GPU
#SBATCH --partition=test                # 使用GPU分区
#SBATCH --mem=64G                      # 为该作业分配64GB内存
#SBATCH --time=6:00:00                # 设置最大运行时间为48小时
#SBATCH --account=EENG026343 		 # 替换为你的HPC项目代码
#SBATCH --output=./log/eval_learned/train.%j.out     # 标准输出文件
#SBATCH --error=./log/eval_learned/train.%j.err      # 错误输出文件

# cd "${SLURM_SUBMIT_DIR}"
# source activate vc
# module load lang/gcc/9.3.0
# module load lang/cuda/11.1
# export CUDA_VISIBLE_DEVICES=0
# yolo val model=jameslahm/yolov10x data=coco.yaml batch=32 device=0,1
# python train_from_scratch.py
# python examples/train.py -m tinylic -d ../../../dataset/PASCAL/all --epochs 400 -lr 1e-4 --batch-size 4 --cuda --save
# python -m compressai.utils.eval_model checkpoint ../../../dataset/PASCAL/all/test -a tinylic -p ./checkpoints/tinylic/3/checkpoint_best_loss.pth.tar --cuda
echo "Start evaluating..."
echo "Start evaluating q3_fine-tune..."
touch ./voc_results/mse/q3_log.txt
python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/vcod/datasets/VOC/images/test2007 -a tinylic -p /home/englishassignment123/work/baseline/vcod/checkpoints_q3/checkpoint_best_loss_compression.pth.tar --cuda --output-path ./voc_results/mse/q3/ > ./voc_results/mse/q3_log.txt

# echo "Start evaluating q2..."
# python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/checkpoint_q2.pth.tar --cuda --output-path ./voc_results/q2/ > ./voc_results/q2_log.txt

# echo "Start evaluating q3..."
# touch ./voc_results/mse/q3_log.txt
# python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/mse/checkpoint_q3.pth.tar --cuda --output-path ./voc_results/mse/q3/ > ./voc_results/mse/q3_log.txt

# # echo "Start evaluating q4..."
# # python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/checkpoint_q4.pth.tar --cuda --output-path ./voc_results/q4/ > ./voc_results/q4_log.txt

# # echo "Start evaluating q5..."
# # python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/checkpoint_q5.pth.tar --cuda --output-path ./voc_results/q5/ > ./voc_results/q5_log.txt

# echo "Start evaluating q6..."
# touch ./voc_results/mse/q6_log.txt
# python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/mse/checkpoint_q6.pth.tar --cuda --output-path ./voc_results/mse/q6/ > ./voc_results/mse/q6_log.txt

# # echo "Start evaluating q7..."
# # python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/checkpoint_q7.pth.tar --cuda --output-path ./voc_results/q7/ > ./voc_results/q7_log.txt

# echo "Start evaluating q8..."
# touch ./voc_results/mse/q8_log.txt
# python -m compressai.utils.eval_model checkpoint /home/englishassignment123/work/baseline/object-detection/datasets/VOC/images/test2007 -a tinylic -p ./pretrain-weight/mse/checkpoint_q8.pth.tar --cuda --output-path ./voc_results/mse/q8/ > ./voc_results/mse/q8_log.txt

echo "Finish evaluating."