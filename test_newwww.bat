@echo off

setlocal enabledelayedexpansion

rem ----------------------------------
set model_name_list="S2R_256_DQN1_netD1_G_GAN_G_L1"
rem ----------------------------------
rem add DQN, netG_loss_setting "G_GAN+G_L1"

for %%m in (%model_name_list%) do (
    python3 test_RL_newwww.py --dataroot "Y:\111\111033631_Yen\ARM\GAN_images\_010_010_010_shuffle_False_502_36" ^
                    --model pix2pix ^
                    --direction AtoB ^
                    --netG resnet_9blocks ^
                    --crop_size 256 ^
                    --load_size 256 ^
                    --num_test 500 ^
                    --input_nc 1 ^
                    --output_nc 1 ^
                    --name "%%m" ^
                    --is_added_DQN 1
)

endlocal
