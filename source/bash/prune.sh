
cd ..
python3 tools/prune.py \
--dataset="Chapman" --num_classes=4 \
--use_demographic \
--num_gpus=2
python3 tools/prune.py \
--dataset="CPSC-2018" --num_classes=9 \
--multilabel \
--use_demographic \
--num_gpus=2

cd bash/