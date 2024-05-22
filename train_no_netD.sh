#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
n_epochs=150

model_name_list=( \
                 "pix2pix_arm_84_DQN1_netD0_netGloss_GAN_RLL1_wgan" \
                 "pix2pix_arm_84_DQN1_netD0_netGloss_GAN_RLL1_nor_wgan" \
                 "pix2pix_arm_84_DQN1_netD0_netGloss_GAN_RLcos_wgan" \
                 "pix2pix_arm_84_DQN1_netD0_netGloss_GAN_RLcos_nor_wgan" \
                 )
                
netG_loss_list=( \
                "G_L1_RL" \
                "G_L1_RL_nor" \
                "G_cos_RL" \
                "G_cos_RL_nor" \
                )

# gan_mode_list=( \
#             #    "lsgan" \
#                "lsgan" \
#             #    "wgangp" \
#                "wgangp" \
#                )              

for (( i = 1; i <= ${#model_name_list[@]}; i++ )); do
    python train.py --dataroot ./Robotic_arm_image_84/RS \
                    --model pix2pix \
                    --direction AtoB \
                    --netG resnet_9blocks \
                    --crop_size 84 \
                    --load_size 84 \
                    --input_nc 1 \
                    --output_nc 1 \
                    --name "${model_name_list[$i]}" \
                    --is_added_DQN 1 \
                    --netD_existed 0 \
                    --netD "numerical" \
                    --netD_input B \
                    --netG_loss_setting "${netG_loss_list[$i]}" \
                    --n_epochs ${n_epochs} \
                    --n_epochs_decay ${n_epochs} \
                    --gan_mode "wgangp"
done

# # ----------------------------------
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
# # ----------------------------------
# # # add DQN, netD: PN, netG_loss_setting "GAN+G_L1"
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netDPN_netGloss_GAN_L1 \
#                 --netD_existed 1 \
#                 --netD_input B \
#                 --is_added_DQN 1 \
#                 --netD "pixelnumerical" \
#                 --pixelnumerical_type 1 \
#                 --netG_loss_setting "G_GAN+G_L1" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --is_save 1
# # ----------------------------------
# # # add DQN, netD: PN, netG_loss_setting "GAN+G_L1"
# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netDPN_netGloss_GAN_L1_RLL1 \
#                 --netD_existed 1 \
#                 --netD_input B \
#                 --is_added_DQN 1 \
#                 --netD "pixelnumerical" \
#                 --pixelnumerical_type 1 \
#                 --netG_loss_setting "G_GAN+G_L1+G_L1_RL" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --is_save 1
# ----------------------------------
# add DQN, no netD, netG_loss_setting "G_L1_RL"
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
# ----------------------------------
# add DQN, no netD, netG_loss_setting "G_L1+G_L1_RL"
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


# n_epochs=10
# n_epochs_decay=10
# ----------------------------------
# add DQN, no netD, netG_loss_setting "G_L1_RL"
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


# python train.py --dataroot ./Robotic_arm_image_84/RS \
#                 --model pix2pix \
#                 --direction AtoB \
#                 --netG resnet_9blocks \
#                 --crop_size 84 \
#                 --load_size 84 \
#                 --input_nc 1 \
#                 --output_nc 1 \
#                 --name pix2pix_arm_84_DQN1_netD0_netGloss_RLcos_TTTT \
#                 --is_added_DQN 1 \
#                 --netD_existed 0 \
#                 --netG_loss_setting "G_L1_RL" \
#                 --n_epochs ${n_epochs} \
#                 --n_epochs_decay ${n_epochs} \
#                 --netD_input B

                # G_cos_RL

