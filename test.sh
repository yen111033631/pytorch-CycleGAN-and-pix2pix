#!./scripts/train_pix2pix.sh

# ----------------------------------
# original train
# netD_input="AB"
# is_added_DQN=1
# python test_RL.py --dataroot ./Robotic_arm_image_84/RS \
#                   --model pix2pix \
#                   --direction AtoB \
#                   --netG resnet_9blocks \
#                   --name "pix2pix_arm_84_origin_netD_input_${netD_input}" \
#                   --is_added_DQN ${is_added_DQN} \
#                   --crop_size 84 \
#                   --load_size 84 \
#                   --input_nc 1 \
#                   --output_nc 1 

# ----------------------------------
model_name_list=( \
                #  "S2R_256_DQN0_netD1_origin" \
                #  "S2R_256_DQN1_netD1_netGloss_GAN_L1" \
                #  "S2R_256_DQN0_netD1_origin_wgan" \
                #  "S2R_256_DQN1_netD1_netGloss_GAN_L1_wgan" \
                 "S2R_256_arm_trajectory_3090_input3_DQN1_netD1_G_GAN_G_L1" \
                 )
# ----------------------------------
# add DQN, netG_loss_setting "G_GAN+G_L1"
for model_name in $model_name_list
do  
    python test_RL.py --dataroot /home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/_arm_trajectory_010_001_0000_889_889 \
                    --model pix2pix \
                    --direction AtoB \
                    --netG resnet_9blocks \
                    --crop_size 256 \
                    --load_size 256 \
                    --num_test 1000 \
                    --input_nc 3 \
                    --output_nc 1 \
                    --name "${model_name}" \
                    --is_added_DQN 1 \
                    --which_DQN "010"
done                    
