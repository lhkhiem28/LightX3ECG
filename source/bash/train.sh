
cd ..
python tools/train.py \
--dataset="Chapman" --num_classes=4 \
--num_gpus=2
# python tools/train.py \
# --dataset="Chapman" --num_classes=4 \
# --lightweight \
# --num_gpus=2
python tools/train.py \
--dataset="CPSC-2018" --num_classes=9 \
--multilabel \
--num_gpus=2
# python tools/train.py \
# --dataset="CPSC-2018" --num_classes=9 \
# --multilabel \
# --lightweight \
# --num_gpus=2

cd bash/