#!/bin/zsh


# ----------------------------------
# original train
n_epochs=200

# model name
model_prefix="S2R_256_arm"
# data_folder_name="002_003_005_test_006"

# loss setting
# # G_GAN_existed="G_GAN"
# G_L1_list=("G_L1" "")
# gan_loss_list=("G_L1_RL" "G_L1_RL_nor" "G_cos_RL" "")

G_GAN_existed="G_GAN"
G_L1_list=("G_L1")
gan_loss_list=("")

# some setting
is_added_DQN=1       # 0: basic, 1: numerical 
netD_existed=1
input_nc=3
which_DQN="010"

# data_dir="/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/${data_folder_name}"

data_folder_name_list=(002_003_005_a 002_003_005_b 002_003_005_c 002_003_005_d 002_003_005_e \
    002_003_005_006_a 002_003_005_006_b 002_003_005_006_c 002_003_005_006_d 002_003_005_006_e \
    )
data_dir="/home/yen/code/yen/DATA/GAN_images"


# 迴圈處理
for gan_loss in "${gan_loss_list[@]}"; do
    for G_L1 in "${G_L1_list[@]}"; do
        for data_folder_name in "${data_folder_name_list[@]}"; do
            if [[ "$G_L1" == "G_L1" ]]; then
                desh_G_L1="_${G_L1}"
                plus_G_L1="+${G_L1}"
            else
                desh_G_L1=""
                plus_G_L1=""
            fi

            if [[ "$gan_loss" == "" ]]; then
                desh_gan_loss=""
                plus_gan_loss=""
            else
                desh_gan_loss="_${gan_loss}"
                plus_gan_loss="+${gan_loss}"
            fi

            # 拼接 model_name 和 GAN_loss_set
            model_name="${model_prefix}_${data_folder_name}_input${input_nc}_DQN${is_added_DQN}_netD${netD_existed}_${G_GAN_existed}${desh_G_L1}${desh_gan_loss}"
            netG_loss_setting="${G_GAN_existed}${plus_G_L1}${plus_gan_loss}"
            echo "$model_name"
            echo "$netG_loss_setting"
            echo "---------------------------------"
            # python train.py --dataroot "${data_dir}/${data_folder_name}" \
            #                 --model pix2pix \
            #                 --direction AtoB \
            #                 --netG resnet_9blocks \
            #                 --crop_size 256 \
            #                 --load_size 256 \
            #                 --input_nc ${input_nc} \
            #                 --output_nc 1 \
            #                 --name "${model_name}" \
            #                 --is_added_DQN ${is_added_DQN} \
            #                 --which_DQN "${which_DQN}" \
            #                 --netD_existed ${netD_existed} \
            #                 --netD "numerical" \
            #                 --netD_input B \
            #                 --netG_loss_setting "${netG_loss_setting}" \
            #                 --n_epochs ${n_epochs} \
            #                 --n_epochs_decay ${n_epochs} \
            #                 --gan_mode "lsgan"

            
            python test_RL.py --dataroot "${data_dir}/${data_folder_name}" \
                            --model pix2pix \
                            --direction AtoB \
                            --netG resnet_9blocks \
                            --crop_size 256 \
                            --load_size 256 \
                            --num_test 1000 \
                            --input_nc ${input_nc} \
                            --output_nc 1 \
                            --name "${model_name}" \
                            --is_added_DQN 1 \
                            --which_DQN "${which_DQN}"

        done
    done
done


