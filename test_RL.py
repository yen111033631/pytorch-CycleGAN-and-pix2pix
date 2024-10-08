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
import torch
import matplotlib.pyplot as plt

import pandas as pd

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
    txt_list = [f"./results/{opt.name}/test_result.txt", f"./results/{opt.name}/test_latest/test_result.txt"]
    for txt_dir in txt_list:
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
def save_loss_png(loss:list):
    plt.figure(figsize=(10, 6))
    plt.hist(loss, bins=30, alpha=0.75, edgecolor='black')
    plt.title('Distribution of Sample Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"./results/{opt.name}/test_latest/loss.png")

def save_csv(loss:list):
    
    # 將列表轉換為 Pandas Series
    data_series = pd.Series(loss)

    # 存成 CSV 檔案
    data_series.to_csv(f"./results/{opt.name}/test_latest/loss_series.csv", index=False, header=False)
    

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

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        model.compute_loss()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    save_txt(opt, model.losses_list, model.same_list)
    save_loss_png(model.losses_list)    
    save_csv(model.losses_list)
    
    # print(model.losses_list)
    # print(model.same_list)
    print("--------------------------------------")
    print("loss mean: \t", np.mean(model.losses_list), "\n"
          "loss std:  \t", np.std(model.losses_list), "\n"
          "loss max:  \t", np.max(model.losses_list), "\n"
          "loss min:  \t", np.min(model.losses_list), "\n"
          "loss lens: \t", len(model.losses_list), "\n"
          "success rate:\t", np.mean(model.same_list)
          )
    print("--------------------------------------")
    webpage.save()  # save the HTML
