#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
# ----------------------------------
# original test
# python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
# ----------------------------------
# try resnet
python train.py --dataroot ./datasets/facades --name facades_pix2pix_netG_resnet_9blocks --model pix2pix --direction BtoA --netG resnet_9blocks --n_epochs 200 --n_epochs_decay 200
python test.py --dataroot ./datasets/facades --name facades_pix2pix_netG_resnet_9blocks --model pix2pix --direction BtoA --netG resnet_9blocks
# ----------------------------------
# try 
# python train.py --dataroot ./datasets/facades --name facades_pix2pix_test --model pix2pix --direction BtoA --netG resnet_9blocks --n_epochs 1 --n_epochs_decay 0
# python test.py --dataroot ./datasets/facades --name facades_pix2pix_1 --model pix2pix --direction BtoA --netG resnet_9blocks
# ----------------------------------