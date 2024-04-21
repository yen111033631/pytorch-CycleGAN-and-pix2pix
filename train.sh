#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_origin_netD_input_B --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 0 --n_epochs 150 --n_epochs_decay 150 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_origin_netD_input_AB --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 0 --n_epochs 150 --n_epochs_decay 150 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input AB
# ----------------------------------
# netG loss setting
python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_test --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 1 --netG_loss_setting  --n_epochs 10 --n_epochs_decay 0 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B







# original test
# python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
# ----------------------------------
# try resnet
# python train.py --dataroot ./datasets/facades --name facades_pix2pix_netG_resnet_9blocks --model pix2pix --direction BtoA --netG resnet_9blocks --n_epochs 200 --n_epochs_decay 200
# python test.py --dataroot ./datasets/facades --name facades_pix2pix_netG_resnet_9blocks --model pix2pix --direction BtoA --netG resnet_9blocks
# ----------------------------------
# try 
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_test --model pix2pix --direction AtoB --netG resnet_9blocks --n_epochs 1 --n_epochs_decay 0 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input "B"
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_test --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN True --n_epochs 1 --n_epochs_decay 0 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B

# ----------------------------------
# add RL
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_add_RL --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 1 --n_epochs 100 --n_epochs_decay 100 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_add_RL___ --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 1 --n_epochs 200 --n_epochs_decay 200 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_no_DQN --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 0 --n_epochs 150 --n_epochs_decay 150 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B
## test add RL
# python test_RL.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_add_RL --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 0  --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84
# ----------------------------------
# add RL no D
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_add_RL_no_D --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 1 --n_epochs 150 --n_epochs_decay 150 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B --netD_existed 0
# python train.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_add_RL_no_D_L1_and_RL_L1 --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 1 --n_epochs 150 --n_epochs_decay 150 --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84 --netD_input B --netD_existed 0
## test add RL
# python test_RL.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_add_RL --model pix2pix --direction AtoB --netG resnet_9blocks --is_added_DQN 0  --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84
# python test_RL.py --dataroot ./Robotic_arm_image_84/RS --name pix2pix_arm_84_gray_netD_nocat --model pix2pix --direction BtoA --netG resnet_9blocks --is_added_DQN 0  --crop_size 84 --input_nc 1 --output_nc 1 --load_size 84

