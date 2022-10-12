##
# C) scaling for e2cnn

python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=1000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=2000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=3000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=6000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=12000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=24000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=48000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=96000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=128000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=144000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=168000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=180000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn data.size=192000

python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=1000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=2000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=3000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=6000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=12000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=24000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=48000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=96000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=128000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=144000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=168000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=180000
python train_sgd_variant.py name=sgd_noaug hyp=base_da data=CINIC10 model=e2cnn18 data.size=192000


# and InvariantResNet
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=1000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=2000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=3000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=6000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=12000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=24000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=48000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=96000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=128000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=144000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=168000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=180000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=flipinvariantresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=192000


# and OrbitResNet
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=1000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=2000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=3000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=6000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=12000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=24000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=48000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=96000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=128000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=144000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=168000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=180000
python train_sgd_variant.py name=sgd_hflip hyp=base_da data=CINIC10 model=orbitresnet +data.augmentations_train={RandomHorizontalFlip:0.5} data.size=192000
