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
                 "pix2pix_arm_84_origin_netD_input_B" \
                 "pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1" \
                 "pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1_RLL1" \
                 "pix2pix_arm_84_DQN1_netD0_netGloss_RLL1" \
                 "pix2pix_arm_84_DQN1_netD0_netGloss_L1_RLL1" \
                 "pix2pix_arm_84_DQN1_netDPN_netGloss_GAN_L1" \
                 "pix2pix_arm_84_DQN1_netDPN_netGloss_GAN_L1_RLL1" \
                 )
# ----------------------------------
# add DQN, netG_loss_setting "G_GAN+G_L1"
for model_name in $model_name_list
do 
    # python test_RL.py --dataroot ./Robotic_arm_image_84/RS \
    #                 --model pix2pix \
    #                 --direction AtoB \
    #                 --netG resnet_9blocks \
    #                 --crop_size 84 \
    #                 --load_size 84 \
    #                 --input_nc 1 \
    #                 --output_nc 1 \
    #                 --name "${model_name}" \
    #                 --is_added_DQN 0 
    python test_RL.py --dataroot ./Robotic_arm_image_84/RS \
                    --model pix2pix \
                    --direction AtoB \
                    --netG resnet_9blocks \
                    --crop_size 84 \
                    --load_size 84 \
                    --input_nc 1 \
                    --output_nc 1 \
                    --name "${model_name}" \
                    --is_added_DQN 1 
done                    

                #   --name pix2pix_arm_84_origin_netD_input_B \
                #   --name pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1 \
                # --name pix2pix_arm_84_DQN1_netD1_netGloss_GAN_L1_RLL1 \
                # --name pix2pix_arm_84_DQN1_netD0_netGloss_RLL1 \
                # --name pix2pix_arm_84_DQN1_netD0_netGloss_L1_RLL1 \
