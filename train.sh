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

# for (( i = 1; i <= ${#model_name_list[@]}; i++ )); do
#     python train.py --dataroot ./Robotic_arm_image_84/RS \
#                     --model pix2pix \
#                     --direction AtoB \
#                     --netG resnet_9blocks \
#                     --crop_size 84 \
#                     --load_size 84 \
#                     --input_nc 1 \
#                     --output_nc 1 \
#                     --name "${model_name_list[$i]}" \
#                     --is_added_DQN ${is_added_DQN_list[$i]} \
#                     --netD_existed ${netD_existed_list[$i]} \
#                     --netD "${netD_setting_list[$i]}" \
#                     --netD_input B \
#                     --netG_loss_setting "${netG_loss_list[$i]}" \
#                     --n_epochs ${n_epochs} \
#                     --n_epochs_decay ${n_epochs} \
#                     --gan_mode "${gan_mode_list[$i]}"
# done



# ----------------------------------
# try something
model_name="pix2pix_arm_256_DQN1_netD1_try"
n_epochs=1

image_size=84

python train.py --dataroot ./Robotic_arm_image_256/RS \
               --model pix2pix \
               --direction AtoB \
               --netG resnet_9blocks \
               --crop_size ${image_size} \
               --load_size ${image_size} \
               --input_nc 1 \
               --output_nc 1 \
               --name "${model_name}" \
               --is_added_DQN 1 \
               --netD_existed 1 \
               --netD "numerical" \
               --netD_input B \
               --netG_loss_setting "G_GAN+G_L1" \
               --n_epochs ${n_epochs} \
               --n_epochs_decay ${n_epochs} \
               --gan_mode "lsgan"
