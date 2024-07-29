"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os, glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
from util.others import *
import numpy as np
from torchvision import transforms
import cv2, torch
from PIL import Image
import rs
import time
import pandas as pd
import threading

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

setup_seed(525)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    DRV_chain = read_urdf("DRV90.urdf")
    
    if opt.eval:
        model.eval()
    
    # =======================================================================================
    my_cam = rs.Cam()
    Mem = rs.MemoryCommunicator()
    address = 0x1100
    
    csv_dir = r"\\140.114.141.95\nas\111\111033631_Yen\ARM\capture_images_sim\Jul16_H14_M43_S14_010_0100_882_882\cube_points__.csv"
    # csv_dir = r"\\140.114.141.95\nas\111\111033631_Yen\ARM\capture_images_sim\cube_points__.csv"
    csv_name = os.path.basename(os.path.dirname(csv_dir))
    df = pd.read_csv(csv_dir)
    
    # 創建一個事件對象
    stop_event = threading.Event()

    # 創建並啟動線程
    thread_cam = threading.Thread(target=get_frame, args=(my_cam, stop_event,))
    thread_cam.start()
    time.sleep(1)

    
    # tem joint
    j = [0] * 6
    
    success_list = []
    for k in range(50):
        # ----------------------------------------------------------------
        # set cube position
        cube_position = df.iloc[k+10]
        cube_position__ = [x * 1000 for x in cube_position]
        print("cube_position", cube_position)
        p = [*cube_position__[:2], 100]
        
        # -----------------
        # check ik
        cube_p = cube_position.copy()
        cube_p[-1] = 0.1
        is_flip = check_wrist_flip(DRV_chain, cube_p)
        # -----------------        
        
        # write data into memory
        Mem.write_data([2, *p, is_flip], address)
        detect_y()
        Mem.write_data([3], address)
        time.sleep(2)
        # ----------------------------------------------------------------
        i = 0
        while True:
            # ------------------------------------------------------------
            # get frame
            frame = my_cam.color_image
            # ------------------------------------------------------------
            # transfer frame into tensor
            start = time.time()
            # transfer cv2 to PIL
            image = cv2_to_pil(frame)
            # get tensor
            image_tensor = get_tensor(image, size=256)
            # ------------------------------------------------------------
            # input tensor into model, output fake_B and displacement
            fake_B_tensor, displacement = model.S2R_displacement(image_tensor) 
            fake_B_img = util.tensor2im(fake_B_tensor.cpu())
            # print(displacement)
            end = time.time()
            print("spend time:", round(end-start, 5))
            
            # ------------------------------------------------------------
            # show image
            # cv2.imshow('RealSense', frame)
            cv2.imshow('fake_B_img', fake_B_img)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()

            # ------------------------------------------------------------
            # get next position
            position = Mem.read_data(3, address=0x00F0)
            next_position = position + displacement       
            
            # if i == 0:
            #     next_position = cube_position.copy()
            #     next_position[-1] = next_position[-1] + 0.12
            #     print("-")
            #     print("cube_position", cube_position)
            #     print("next_position", next_position)
            # ------------------------------------------------------------
            # check if next position is safe
            print(i, next_position)
            if not(check_next_position_is_safe(next_position)):
                is_success = 0
                print("not safe")
                Mem.write_data([3], address)
                time.sleep(.01)
                break
            
            is_success = check_is_success(next_position, cube_position)            
            # ------------------------------------------------------------
            # write data into memory            
            next_position__ = [x * 1000 for x in next_position]            
            Mem.write_data([1, *next_position__, check_wrist_flip(DRV_chain, next_position), is_success], address)
            time.sleep(0.01)
            if is_success:
                while True:
                    data = Mem.read_data(1, address=0x1200)
                    if data[0] == 87: break   
                    time.sleep(1)
                break                         
            # ------------------------------------------------------------
            # check if exceed max step
            if i > 90:
                break
            # ------------------------------------------------------------
            i += 1
        # ----------------------------------------------------------------
        # move arm back
        print("out")
        Mem.write_data([3], address)
        while True:
            data = Mem.read_data(1, address=address)
            if data[0] == 10: break
        # ----------------------------------------------------------------
        # record success or not
        print(is_success)
        success_list.append(is_success)
    # --------------------------------------------------------------------
    # end of all episode
    print(success_list)
    print(sum(success_list) / len(success_list))
    
    Mem.write_data([-1], address)
    time.sleep(0.1)
        
    # 發送停止信號
    print("Sending stop signal...")
    stop_event.set()        
    # cv2.destroyAllWindows()
    # --------------------------------------------------------------------