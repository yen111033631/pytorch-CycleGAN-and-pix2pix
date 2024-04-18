#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
# ----------------------------------
# original test
# python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
# ----------------------------------
# try resnet
# python train.py --dataroot ./datasets/facades --name facades_pix2pix_netG_resnet_9blocks --model pix2pix --direction BtoA --netG resnet_9blocks --n_epochs 200 --n_epochs_decay 200
# python test.py --dataroot ./datasets/facades --name facades_pix2pix_netG_resnet_9blocks --model pix2pix --direction BtoA --netG resnet_9blocks
# ----------------------------------
# try 
python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_test --model pix2pix --direction AtoB --netG resnet_9blocks --n_epochs 1 --n_epochs_decay 0 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input "B"
# python train.py --dataroot ./datasets/facades --name facades_pix2pix_256_test --model pix2pix --direction BtoA --netG resnet_9blocks --n_epochs 1 --n_epochs_decay 0 --input_nc 1 --output_nc 1 --netD_input "AB"



# ----------------------------------
# gray 
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_gray --model pix2pix --direction AtoB --netG resnet_9blocks --n_epochs 100 --n_epochs_decay 100 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84
# python test.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_gray --model pix2pix --direction AtoB --netG resnet_9blocks  --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84

# ----------------------------------
# gray net_D no concat 
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_gray_netD_nocat --model pix2pix --direction AtoB --netG resnet_9blocks --n_epochs 100 --n_epochs_decay 100 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input "B"
# python test.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_gray --model pix2pix --direction AtoB --netG resnet_9blocks  --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84


# ----------------------------------
# arm
# python train.py --dataroot ./Robotic_arm_image/RS --name facades_pix2pix_arm --model pix2pix --direction AtoB --netG resnet_9blocks --n_epochs 100 --n_epochs_decay 100


# ----------------------------------
# 256_BtoA
# python train.py --dataroot ./Robotic_arm_image_256/RS --name pix2pix_arm_256_BtoA --model pix2pix --direction BtoA --netG resnet_9blocks --n_epochs 200 --n_epochs_decay 200
# python train.py --dataroot ./Robotic_arm_image_256/RS --name pix2pix_arm_256_AtoB --model pix2pix --direction AtoB --netG resnet_9blocks --n_epochs 200 --n_epochs_decay 200

# python test.py --dataroot ./Robotic_arm_image_256/RS --name pix2pix_arm_256_BtoA --model pix2pix --direction BtoA --netG resnet_9blocks
# python test.py --dataroot ./Robotic_arm_image_256/RS --name pix2pix_arm_256_AtoB --model pix2pix --direction AtoB --netG resnet_9blocks
