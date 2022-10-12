##


# A) model param scaling for TrivialAug&Flips&Crops
# base model
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
# width 4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=4


# width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=8
# model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=16
# model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=32
# model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=96
# model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=128
# model.width=256
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=256 hyp.sub_batch=64
# # model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=512
# # model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=196
# model.width=384
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5} model.width=384 hyp.sub_batch=64

#####################################
# also need all of these with _noaug
#####################################
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=4
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=4


# width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=8
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=8
# model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=16
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=16
# model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=32
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=32
# model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=96
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=96
# model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=128
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=128
# model.width=256
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=256 hyp.sub_batch=64
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=256 hyp.sub_batch=64
# # model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=512
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=512
# # model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=196
python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=196
# model.width=384
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=1000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=2000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=3000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=6000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=12000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=24000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=48000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=96000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=128000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=144000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=168000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=180000 model.width=384 hyp.sub_batch=64
# python train_sgd_variant.py name=sgd_noaug_rand hyp=base_da data=CINIC10 data.size=192000 model.width=384 hyp.sub_batch=64


# same for hflip later
