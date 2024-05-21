#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
n_epochs=150

model_name_list=( \
                #  "pix2pix_arm_84_DQN0_netD1_origin" \
                 "pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1" \
                #  "pix2pix_arm_84_DQN0_netD1_origin_wgan" \
                 "pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1_wgan" \
                 )

is_added_DQN_list=(\
                #    0 \
                   1 \
                #    0 \
                   1 \
                   )    

netD_existed_list=(\
                   1 \
                   1 \
                   1 \
                   1 \
                   )  

netD_setting_list=( \
                #    "basic" \
                   "numerical" \
                #    "basic" \
                   "numerical" \
                   )                     

                
netG_loss_list=( \
                "G_GAN+G_L1" \
                "G_GAN+G_L1" \
                "G_GAN+G_L1" \
                "G_GAN+G_L1" \
                )

gan_mode_list=( \
            #    "lsgan" \
               "lsgan" \
            #    "wgangp" \
               "wgangp" \
               )              

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
                    --is_added_DQN ${is_added_DQN_list[$i]} \
                    --netD_existed ${netD_existed_list[$i]} \
                    --netD "${netD_setting_list[$i]}" \
                    --netD_input B \
                    --netG_loss_setting "${netG_loss_list[$i]}" \
                    --n_epochs ${n_epochs} \
                    --n_epochs_decay ${n_epochs} \
                    --gan_mode "${gan_mode_list[$i]}"
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

