#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
netD_input="AB"
is_added_DQN=1
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_origin_netD_input_B \
#                 --is_added_DQN 0 \
#                 --n_epochs 150 \
#                 --n_epochs_decay 150 \
#                 --netD_input B

# python test_RL.py --dataroot ./Robotic_arm_image_84/RS \
#                   --name "pix2pix_arm_84_origin_netD_input_${netD_input}" \
#                   --model pix2pix \
#                   --direction AtoB \
#                   --netG resnet_9blocks \
#                   --is_added_DQN ${is_added_DQN} \
#                   --crop_size 84 \
#                   --load_size 84 \
#                   --input_nc 1 \
#                   --output_nc 1 

# ----------------------------------
n_epochs=150
n_epochs_decay=150
# ----------------------------------
# # add DQN, netG_loss_setting "G_GAN+G_L1"
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1 \
#                 --is_added_DQN 1 \
#                 --netD_existed 1 \
#                 --netG_loss_setting "G_GAN+G_L1" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --netD_input B
# # ----------------------------------
# # add DQN, netG_loss_setting "G_GAN+G_L1+G_L1_RL"
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1_RLL1 \
#                 --is_added_DQN 1 \
#                 --netD_existed 1 \
#                 --netG_loss_setting "G_GAN+G_L1+G_L1_RL" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --netD_input B

# # ----------------------------------
# # add DQN, no netD, netG_loss_setting "G_L1_RL"
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netD0_netGloss_RLL1 \
#                 --is_added_DQN 1 \
#                 --netD_existed 0 \
#                 --netG_loss_setting "G_L1_RL" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --netD_input B
# # ----------------------------------
# # add DQN, no netD, netG_loss_setting "G_L1+G_L1_RL"
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netD0_netGloss_L1_RLL1 \
#                 --is_added_DQN 1 \
#                 --netD_existed 0 \
#                 --netG_loss_setting "G_L1+G_L1_RL" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --netD_input B

n_epochs=1
n_epochs_decay=0
# ----------------------------------
# try a lot of things
python train.py --dataroot ./Robotic_arm_image_84/RS \
                --model pix2pix \
                --direction AtoB \
                --netG resnet_9blocks \
                --crop_size 84 \
                --load_size 84 \
                --input_nc 1 \
                --output_nc 1 \
                --name pix2pix_arm_84_DQN1_testttt \
                --netD_existed 1 \
                --netD_input B \
                --is_added_DQN 1 \
                --netD "pixelnumerical" \
                --pixelnumerical_type 1 \
                --netG_loss_setting "G_GAN+G_L1" \
                --n_epochs ${n_epochs} \
                --n_epochs_decay ${n_epochs} \
                --is_save 0

                # netD basic, numerical, pixelnumerical
                # netG_loss_setting "G_GAN+G_L1+G_L1_RL" 