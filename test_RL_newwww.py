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
    thread1 = threading.Thread(target=get_frame, args=(my_cam, stop_event,))
    thread1.start()
    time.sleep(1)

    
    
    
    success_list = []
    for k in range(10):
        cube_position = df.iloc[k]
        cube_position__ = [x * 1000 for x in cube_position]
        print("cube_position", cube_position)
        p = [*cube_position__[:2], 100]
        
        j = [0] * 6

        Mem.write_data([2, *j, *p], address)
        detect_y()
        Mem.write_data([3], address)
        time.sleep(2)
        
        # 創建一個事件對象
        success_event = threading.Event()
        pause_event = threading.Event()
        target_position = cube_position.copy()
        target_position[-1] = target_position[-1] + 0.1
        success_distance = 0.05

        # 創建並啟動線程
        thread2 = threading.Thread(target=detect_success, args=(success_event, pause_event, target_position, success_distance, Mem,))
        thread2.start()        
        time.sleep(2)

        
        i = 0
        while not success_event.is_set():
            # get frame
            frame = my_cam.color_image

            # position = get_current_position()
            # position = Mem.read_data(3, address=0x00F0)
            # position = [1, 2, 3]
            # print(position)

            start = time.time()
            # transfer cv2 to PIL
            image = cv2_to_pil(frame)
            # get tensor
            image_tensor = get_tensor(image, size=256)
            fake_B_tensor, displacement = model.S2R_displacement(image_tensor) 
            fake_B_img = util.tensor2im(fake_B_tensor.cpu())
            # print(displacement)
            end = time.time()
            print("spend time:", round(end-start, 5))
            
            # cv2.imshow('RealSense', frame)
            cv2.imshow('fake_B_img', fake_B_img)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()

            position = Mem.arm_position
            next_position = position + displacement
            
            # if i == 1:
            #     # next_position = [0.24,-0.3, 0.199]
            #     next_position = [0.112151, -0.378836, 0.18000699]
            
            print(i, next_position)
            if not(check_next_position_is_safe(next_position)):
                is_success = -1
                print("not safe")
                pause_event.set()
                time.sleep(.2)
                Mem.write_data([3], address)
                time.sleep(.01)
                pause_event._flag = False
                success_event.set()
                break
            
            # is_success = check_is_success(next_position, cube_position)
            # if is_success: print("aha")
            
            next_position__ = [x * 1000 for x in next_position]

            j = [0] * 6

            pause_event.set()
            time.sleep(.2)
            if success_event.is_set():
                Mem.write_data([3], address)
                break
            else:
                Mem.write_data([1, *j, *next_position__], address)
            time.sleep(0.1)
            pause_event._flag = False
            # while True:
            #     data = Mem.read_data(1, address=address)
            #     if data[0] == 10: 
            #         break
            

            
            if i > 100:
                # Mem.write_data([3], address)
                # if is_success: 
                #     print("success!")
                #     Mem.write_data([11, *j, *next_position__], address)
                #     time.sleep(0.1)
                #     while True:
                #         data = Mem.read_data(1, address=address)
                #         if data[0] == 20: 
                #             break
                time.sleep(5)
                break
            i += 1
        pause_event.set()
        Mem.write_data([3], address)
        while True:
            data = Mem.read_data(1, address=address)
            if data[0] == 10: 
                break
        
        pause_event._flag = False
            
        print("out")
        is_success = 1 if success_event.is_set() else is_success
        success_list.append(is_success)
    print(success_list)
    print(sum(success_list) / len(success_list))
    
    Mem.write_data([-1], address)
    time.sleep(0.1)
        
        
    # 發送停止信號
    print("Sending stop signal...")
    stop_event.set()        
    # cv2.destroyAllWindows()
        
        
    # =======================================================================================
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/all/test/img_0804.jpg"
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/_010_010_010_shuffle_False_502_36/test/img_0002.jpg"
    # a, b = read_from_PIL(image_path)
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/capture_images_real/Jun17_H15_M21_S56_010_010_010_shuffle_False_502_36_001/img_0000.jpg"
    # # ab = cv2.imread(image_path, 0)
    # # a, b = split_image(ab)
    
    # # transform = get_tensor()
    
    
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/capture_images_real/images/Jul02_H22_M27_S59/img_1_color.bmp"
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/capture_images_real/Jul16_H14_M43_S14_010_0100_882_882_002/img_0000.bmp"
    # image_path = r"\\140.114.141.95\nas\111\111033631_Yen\ARM\capture_images_real\Jul16_H14_M43_S14_010_0100_882_882_002\img_0000.bmp"
    # a = Image.open(image_path)
    # a.save('real_A_img.jpg')
    # a_tensor = get_tensor(a)
    
    # fake_B_tensor, displacement = model.S2R_displacement(a_tensor)  
    # fake_B_img = reverse_transform(fake_B_tensor.cpu())
    
    # cv2.imwrite('fake_B_img.jpg', fake_B_img)
    
    
    # image_path_list = glob.glob(r"\\140.114.141.95\nas\111\111033631_Yen\ARM\GAN_images\_010_0100_882_882\test\*.jpg")
    # folder_name =os.path.basename(os.path.dirname(image_path_list[0]))
    
    # folder_dir = f"./test_newww/{folder_name}__"
    # os.makedirs(folder_dir, exist_ok=True)
    
    # for i, image_path in enumerate(image_path_list):
    #     base_name = os.path.basename(image_path)
    #     print(i, os.path.basename(image_path))
        
        
    #     a, b = read_from_PIL(image_path)
    #     # a = Image.open(image_path)
    #     a.save(f"{folder_dir}/{base_name[:-4]}.png")
        
    #     a_tensor = get_tensor(a)
    #     fake_B_tensor, displacement = model.S2R_displacement(a_tensor)  
    #     fake_B_img = reverse_transform(fake_B_tensor.cpu())
        
    #     cv2.imwrite(f"{folder_dir}/{base_name[:-4]}_fake_B.png", fake_B_img)
    

    
    
    # # fake_B_1 = a_tensor
    # # fake_B_2 = aa_tensor
    # # fake_B_1 = model.real_A.cpu()
    # # fake_B_2 = data["A"].cpu()
    # # fake_B_1 = model.fake_B
    # # fake_B_1 = model.netG(data["A"])
    # # fake_B_2 = model.netG(model.real_A)

    # # comparison_result = torch.eq(fake_B_1, fake_B_2)
    
    # # allclose = torch.allclose(fake_B_1, fake_B_2)
    # # print("-")
    # # print("allclose", allclose)
    # # print("-")    
    
    
    
    
    # # image = cv2.imread(image_path, 0)

    # # a, b = split_image(image)
    
    
    # # with torch.no_grad():
    # #     model.eval()
    # #     fake_B = model.netG(a_tensor)
    # #     # print(fake_B[0, 0, 56, :10])
        
    # #     fake_B_RL = fake_B / 2.0 + 0.5
    # #     fake_B_RL = model.agent.DQN(fake_B_RL)    
    # #     action = fake_B_RL.argmax(1)[0].item()
    # #     # print(fake_B_RL[0, :10])
    # #     # print("-")
    # #     # print("aciton", action)
    # #     # print("-")
    
    
    # # r, t, p = spread_index_into_spherical(action, 
    # #                                         theta_num=8, 
    # #                                         shell_unit_length=0.025)
    # # displacement = spherical_to_cartesian(r, t, p)
    # # print(displacement)  
    
    # # displacement = model.S2R_displacement(a_tensor)  
    # # print(displacement)  
    # # model.eval()
    # # while True:
    # #     frame = get_frame()
    # #     frame_tensor = get_tensor(frame)
    
    # # =======================================================================================
    
    

    # # initialize logger
    # if opt.use_wandb:
    #     wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
    #     wandb_run._label(repo='CycleGAN-and-pix2pix')

    # # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # # test with eval mode. This only affects layers like batchnorm and dropout.
    # # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # if opt.eval:
    #     model.eval()
    # for i, data in enumerate(dataset):
    #     # if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #     if i >= 1:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()           # run inference        
        
    #     model.compute_loss()
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()     # get image paths
    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    
    # # ----------------------------------------------------------------------------
    # # save_txt(opt, model.losses_list, model.same_list)
    # # print(model.losses_list)
    # # print(model.same_list)
    # # print("--------------------------------------")
    # # print("loss mean: \t", np.mean(model.losses_list), "\n"
    # #       "loss std:  \t", np.std(model.losses_list), "\n"
    # #       "loss max:  \t", np.max(model.losses_list), "\n"
    # #       "loss min:  \t", np.min(model.losses_list), "\n"
    # #       "loss lens: \t", len(model.losses_list), "\n"
    # #     #   "success rate:\t", np.mean(model.same_list)
    # #       )
    # # print("--------------------------------------")
    # # webpage.save()  # save the HTML
