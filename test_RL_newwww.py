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
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from torchvision import transforms
import cv2, torch
from PIL import Image
import rs
import time
import glob

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

# ------------------------------------------------------------------
# random seed 
def setup_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

setup_seed(525)
# ------------------------------------------------------------------


def save_txt(opt, losses_list, same_list):
    txt_dir = f"./results/{opt.name}/test_result.txt"
    with open(txt_dir, "a") as txt_file:
        txt_file.write("--------------------------------------\n")
        txt_file.write(f"is_added_DQN: \t{opt.is_added_DQN}\n")
        txt_file.write(f"loss mean:    \t{np.mean(losses_list)}\n")
        txt_file.write(f"loss std:     \t{np.std(losses_list)}\n")
        txt_file.write(f"loss lens:    \t{len(losses_list)}\n")
        txt_file.write(f"loss max:     \t{np.max(losses_list)}\n")
        txt_file.write(f"loss min:     \t{np.min(losses_list)}\n")
        txt_file.write(f"success rate: \t{np.mean(same_list)}\n")
        txt_file.write(f"loss lens: \t\t{len(losses_list)}\n")
        txt_file.write("--------------------------------------\n")

# ----------------------------------------------------------------------------
def split_image(image):
    # image = cv2.imread(image_path, 0)
    a = image[:, :image.shape[0]]
    b = image[:, image.shape[0]:]
    return a, b

def add_black_border_to_square_PIL(image):
    """
    將輸入圖像補上黑邊，使其變成正方形
    :param image: PIL.Image 圖像對象
    :return: 補黑邊後的正方形圖像
    """
    # 獲取原始照片的尺寸
    width, height = image.size
    
    # 計算新照片的邊長（取原始照片的較長邊）
    new_size = max(width, height)
    
    # 創建一個黑色的背景圖像
    new_image = Image.new("RGB", (new_size, new_size), (0, 0, 0))

    # 計算原始照片在新照片中的起始位置
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2

    # 將原始照片粘貼到黑色背景圖像的中央
    new_image.paste(image, (x_offset, y_offset))

    return new_image

def read_from_PIL(image_path):
    AB = Image.open(image_path)
    # # split AB image into A and B
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    return A, B

def get_tensor(image, size=256):
    # 定義變換管道
    transform = transforms.Compose([
        transforms.Lambda(add_black_border_to_square_PIL),
        # transforms.Grayscale(1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])
    return transform(image)

def reverse_transform(transformed_image, original_size=None):
    """
    Reverses the transformations applied to the image and displays it.

    Parameters:
    - transformed_image (torch.Tensor): The transformed image tensor.
    - original_size (tuple): The original size of the image (height, width). Default is None.

    Returns:
    - None
    """
    # Step 1: Remove the extra dimension
    image_tensor = transformed_image.squeeze(0)

    # Step 2: Reverse normalization
    inverse_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5], std=[1 / 0.5]
    )
    image_tensor = inverse_normalize(image_tensor)

    # Step 3: Convert the tensor to a NumPy array
    image_np = image_tensor.squeeze().numpy() * 255  # Scale back to [0, 255]
    image_np = image_np.astype(np.uint8)
    
    return image_np


def cv2_to_pil(cv2_image):
    """
    將 OpenCV 圖像轉換為 PIL 圖像
    :param cv2_image: 使用 OpenCV (cv2) 讀取或處理的 numpy 圖像
    :return: PIL.Image 圖像對象
    """
    # 檢查圖像是否為灰階
    if len(cv2_image.shape) == 2:
        # 灰階圖像直接轉換
        pil_image = Image.fromarray(cv2_image)
    else:
        # 彩色圖像需要轉換顏色通道順序 (從 BGR 轉為 RGB)
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image_rgb)
    
    return pil_image
    
# ----------------------------------------------------------------------------

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
    
    model.eval()

    
    # =======================================================================================
    # my_cam = rs.Cam()
    # Mem = rs.MemoryCommunicator()
    # address = 0x1100
    # i = 0
    # while True:
    #     # get frame
    #     frame = my_cam.get_frame()

    #     cv2.imshow('RealSense', frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        

    #     # position = get_current_position()
    #     position = Mem.read_data(3, address=0x00F0)
    #     print(position)

    #     # transfer cv2 to PIL
    #     image = cv2_to_pil(frame)
    #     # get tensor
    #     image_tensor = get_tensor(image, size=256)
    #     fake_B_tensor, displacement = model.S2R_displacement(image_tensor) 
    #     fake_B_img = reverse_transform(fake_B_tensor.cpu())
    #     print(displacement)
        
    #     cv2.imshow('fake_B_img', fake_B_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     next_position = position + displacement
    #     next_position = [x * 1000 for x in next_position]
    #     print(next_position)

    #     j = [0] * 6

    #     Mem.write_data([1, *j, *next_position], address)
    #     time.sleep(0.1)

        
    #     if i > 10:
    #         break
    #     i += 1
    # # =======================================================================================
    # # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/all/test/img_0804.jpg"
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/_010_010_010_shuffle_False_502_36/test/img_0002.jpg"
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/GAN_images/all_002/test/img_0804.jpg"
    # a, b = read_from_PIL(image_path)
    
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/capture_images_real/Jun17_H15_M21_S56_010_010_010_shuffle_False_502_36_001/img_0000.jpg"
    # image_path = "/home/yen/mount/nas/111/111033631_Yen/ARM/capture_images_real/images/Jul02_H22_M27_S59/img_1_color.bmp"
    
    image_path_list = glob.glob("/home/yen/mount/nas/111/111033631_Yen/ARM/capture_images_real/images/Jul04_H22_M15_S40/img_*_color.bmp")
    
    folder_name =os.path.basename(os.path.dirname(image_path_list[0]))
    
    folder_dir = f"./test_newww/{folder_name}__"
    os.makedirs(folder_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_path_list):
        base_name = os.path.basename(image_path)
        print(i, os.path.basename(image_path))
        a = Image.open(image_path)
        a.save(f"{folder_dir}/{base_name[:-4]}.png")
        
        a_tensor = get_tensor(a)
        fake_B_tensor, displacement = model.S2R_displacement(a_tensor)  
        fake_B_img = reverse_transform(fake_B_tensor.cpu())
        
        cv2.imwrite(f"{folder_dir}/{base_name[:-4]}_fake_B.png", fake_B_img)

    
    
    # # # # ab = cv2.imread(image_path, 0)
    # # # # a, b = split_image(ab)
    
    # # # # transform = get_tensor()
    

    
    
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
    #     print(data["A_paths"][0] == image_path)
    #     print(data["A"].shape)
    #     print(a_tensor.shape)
        
    #     tensor_1 = data["A"].cpu()
    #     tensor_2 = a_tensor
        
    #     comparison_result = torch.eq(tensor_1, tensor_2)
    #     print(comparison_result)
        
    #     allclose = torch.allclose(tensor_1, tensor_2)
    #     print("-")
    #     print("allclose", allclose)
    #     print("-")  
        
    #     # 找到等於 False 的位置
    #     false_positions = torch.nonzero(~comparison_result)

    #     # 計算 False 的數量
    #     num_false = (~comparison_result).sum()

    #     print("比較結果:")
    #     print(comparison_result)

    #     print("\n等於 False 的位置:")
    #     print(false_positions)

    #     print("\nFalse 的數量:")
    #     print(num_false)        
        
        
        
    #     fake_B_img_model = reverse_transform(model.fake_B.cpu())
    
    #     cv2.imwrite('fake_B_img_model.png', fake_B_img_model)
         
        
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
