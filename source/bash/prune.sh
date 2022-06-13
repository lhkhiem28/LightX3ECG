
cd ..
python3 tools/prune.py \
--num_gpus=2 \
--dataset="downstream/Chapman" --num_classes=4 \
--lightweight
python3 tools/prune.py \
--num_gpus=2 \
--dataset="downstream/CPSC-2018" --num_classes=9 \
--multilabel \
--lightweight

cd bash/