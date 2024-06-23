#!/bin/zsh


# ----------------------------------
# original train
n_epochs=150

# model name
model_prefix="S2R_256"

# loss setting
G_GAN_existed="G_GAN"
G_L1_list=("G_L1" "")
gan_loss_list=("G_L1_RL" "G_L1_RL_nor" "G_cos_RL" "")

# some setting
is_added_DQN=1
netD_existed=1

data_dir="/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/all"


# 迴圈處理
for gan_loss in "${gan_loss_list[@]}"; do
    for G_L1 in "${G_L1_list[@]}"; do

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
        model_name="${model_prefix}_DQN${is_added_DQN}_netD${netD_existed}_${G_GAN_existed}${desh_G_L1}${desh_gan_loss}"
        netG_loss_setting="${G_GAN_existed}${plus_G_L1}${plus_gan_loss}"
        echo "$model_name"
        echo "$netG_loss_setting"
        echo "---------------------------------"
        python train.py --dataroot "${data_dir}" \
                        --model pix2pix \
                        --direction AtoB \
                        --netG resnet_9blocks \
                        --crop_size 256 \
                        --load_size 256 \
                        --input_nc 1 \
                        --output_nc 1 \
                        --name "${model_name}" \
                        --is_added_DQN ${is_added_DQN} \
                        --netD_existed ${netD_existed} \
                        --netD "numerical" \
                        --netD_input B \
                        --netG_loss_setting "${netG_loss_setting}" \
                        --n_epochs ${n_epochs} \
                        --n_epochs_decay ${n_epochs} \
                        --gan_mode "lsgan"

        
        python test_RL.py --dataroot "${data_dir}" \
                        --model pix2pix \
                        --direction AtoB \
                        --netG resnet_9blocks \
                        --crop_size 256 \
                        --load_size 256 \
                        --input_nc 1 \
                        --output_nc 1 \
                        --name "${model_name}" \
                        --is_added_DQN 1 
    done
done


