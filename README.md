# DBSN
Code for Deep Bayesian Structure Networks

train on classification tasks:
```
cd dbsn

run: random
CUDA_VISIBLE_DEVICES=0 python train.py --run random_0 --dataset cifar10 --method random

run: dropout
CUDA_VISIBLE_DEVICES=0 python train.py --run ds_dp0.2_0 --dataset cifar10 --method densenet --drop_rate 0.2

run: fixed alpha
CUDA_VISIBLE_DEVICES=0 python train.py --run ds_0 --dataset cifar10 --method densenet

run: drop-path
CUDA_VISIBLE_DEVICES=0 python train.py --run ds_dpth0.3_0 --dataset cifar10 --method densenet --droppath_rate 0.3

run: dbsn-1
CUDA_VISIBLE_DEVICES=0 python train.py --run adags_lr3_decayto0.5_1gpu_0

run: dbsn
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --run adags_lr3_decayto0.5_0

run: dbsn*
CUDA_VISIBLE_DEVICES=0 python train.py --run adags_lr3_con1_1gpu_0 (change L63 of architect.py)

run: pe
CUDA_VISIBLE_DEVICES=0 python train.py --run ps_0 --dataset cifar10 --ps

run:darts
CUDA_VISIBLE_DEVICES=0 python train.py --run darts_0 --dataset cifar10 --ps --valid
```

search arch with trained weights:
```
CUDA_VISIBLE_DEVICES=0 python search-arch.py --restore ../work/runadags_lr3_decayto0.5_0/epoch100.pth  --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python search-arch.py --restore ../work/runrandom_0/epoch100.pth  --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python train.py --run random_safrom_decayto0.5 --dataset cifar10 --method random --restore_arch ../work/saadags_lr3_decayto0.5_0/alphas20.pth
CUDA_VISIBLE_DEVICES=0 python train.py --run random_safrom_random --dataset cifar10 --method random --restore_arch ../work/sarandom_0/alphas20.pth
```

train and test nek-fac (based on the [original implementation](https://github.com/pomonam/NoisyNaturalGradient)):
```
cd nng

CUDA_VISIBLE_DEVICES=0 python train.py --config config/classification/ekfac_vgg16_bn_aug.json
vgg11:
80 test cifari10: test | loss: 0.552116 | accuracy: 0.900800
80 test cifari100: test | loss: 2.951545 | accuracy: 0.610300

vgg16 (use this):
100 test cifari10: test | loss: 0.454479 | accuracy: 0.925700 | ece: 0.043436
100 test cifari100: test | loss: 2.784184 | accuracy: 0.625300 | ece: 0.166482

vgg13:
100 test cifari10: test | loss: 0.445817 | accuracy: 0.921800 | ece: 0.055207
100 test cifari100: test | loss: 2.729095 | accuracy: 0.623800 | ece: 0.250805
```

segmentation results:

`cd dbsn_seg`

`CUDA_VISIBLE_DEVICES=0 python train.py --model FCDenseNet67 --dir baseline_pw0.1`
> baseline_pw0.1 (3463243): Test - Loss: 0.3258 | Acc: 0.9040 | IOU: 0.6306 (0.92879747 0.80222609 0.29405931 0.94646308 0.82097352 0.75199645 0.40112767 0.32157756 0.7888852  0.48912112 0.39160665)


`CUDA_VISIBLE_DEVICES=0 python train.py --model DBSN --dir dbsn_bn_pw0.1_clip_3_1gpu`
> dbsn_bn_pw0.1_clip_3_1gpu (3319451): Test - Loss: 0.2822 | Acc: 0.9143 | IOU: 0.6538(800epoch)

> 100times: Test - Loss: 0.2759 | Acc: 0.9148 | IOU: 0.6556 (0.92821956 0.82942139 0.31693728 0.94421578 0.81509191 0.77227186 0.42254635 0.34969423 0.79376511 0.5721504  0.46677926)
