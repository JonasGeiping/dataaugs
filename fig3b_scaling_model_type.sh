##


# A) model param scaling for TrivialAug&Flips&Crops

python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}

python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=convmixer data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} hyp.sub_batch=64

python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=vgg13 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}

python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 model=swinv2 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}

#####################################
# also need all of these with _noaug
#####################################

python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=1000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=2000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=3000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=6000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=12000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=24000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=48000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=96000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=128000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=144000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=168000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=180000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=pyramidnet110 data.size=192000

python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=1000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=2000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=3000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=6000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=12000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=24000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=48000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=96000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=128000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=144000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=168000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=180000 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=convmixer data.size=192000 hyp.sub_batch=64

python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=1000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=2000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=3000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=6000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=12000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=24000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=48000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=96000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=128000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=144000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=168000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=180000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=vgg13 data.size=192000

python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=1000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=2000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=3000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=6000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=12000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=24000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=48000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=96000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=128000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=144000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=168000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=180000
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 model=swinv2 data.size=192000



# same for hflip later
