
cd ..
python3 tools/train.py \
--num_gpus=2 \
--dataset="Chapman" --num_classes=4 \
# --use_demographic
python3 tools/train.py \
--num_gpus=2 \
--dataset="CPSC-2018" --num_classes=9 \
--multilabel \
# --use_demographic

cd bash/