##

# TrivialAug&Crop&Flip
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_trivialaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.TrivialAugmentWide=31 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}


# Crop & Flip
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}

# only flip
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_hflip_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train={RandomHorizontalFlip:0.5}

# only cutout
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.Cutout=[16]
python train_sgd_variant.py name=sgd_cutout_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.Cutout=[16]

# only perspective
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train={RandomPerspective:0.5}
python train_sgd_variant.py name=sgd_perspective_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train={RandomPerspective:0.5}

# cutout & hflip & rcrop
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_cutout_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.Cutout=[16] +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}

# randaug&hflip&rcrop
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=1000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=2000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=3000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=6000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=12000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=24000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=48000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=96000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=128000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=144000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=168000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=180000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
python train_sgd_variant.py name=sgd_randaug_hflip_rcrop_rand hyp=base_da data=CINIC10 data.size=192000 +data.augmentations_train.tvRandAugment=CIFAR10 +data.augmentations_train.RandomCrop=[32,4] +data.augmentations_train={RandomHorizontalFlip:0.5}
