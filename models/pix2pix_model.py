import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
# import .read_DQN
from .read_DQN import set_up_agent, get_q_values
import util.util as util

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # ------------------------------------------------------------------
        # DQN model 
        self.is_added_DQN = opt.is_added_DQN
        # print("self.is_added_DQN", self.is_added_DQN == True)
        self.agent = set_up_agent()
        opt.netD = "numerical" if self.is_added_DQN else opt.netD
        # ------------------------------------------------------------------
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_input = opt.netD_input
            if self.netD_input == "AB":
                netD_input_nc = opt.input_nc + opt.output_nc
            elif self.netD_input == "B":
                netD_input_nc = opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        # # ------------------
        # print(self.netD)
        # # ------------------
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        
    def input_RL_model(self):
        # ----------------------------------------------------------------------------
        # add DQN
        if self.is_added_DQN:
            # print(self.is_added_DQN)

            self.fake_B_RL = self.fake_B / 2.0 + 0.5
            self.fake_B_RL = self.agent.DQN(self.fake_B_RL)
            # self.fake_B_RL = self.fake_B_RL / 100

            self.real_B_RL = self.real_B / 2.0 + 0.5
            self.real_B_RL = self.agent.DQN(self.real_B_RL)
            # self.real_B_RL = self.real_B_RL / 100
        # ----------------------------------------------------------------------------
        

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.netD_input == "AB":
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        elif self.netD_input == "B":
            fake_AB = self.fake_B  if not(self.is_added_DQN)  else self.fake_B_RL  # we use conditional GANs; we need to feed output to the discriminator
        
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
       
        # Real
        if self.netD_input == "AB":
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        elif self.netD_input == "B":
            real_AB = self.real_B if not(self.is_added_DQN)  else self.real_B_RL     
        
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.netD_input == "AB":
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        elif self.netD_input == "B":
            fake_AB = self.fake_B if not(self.is_added_DQN)  else self.fake_B_RL 

        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        if self.is_added_DQN:
            Fake_B = self.fake_B_RL
            Real_B = self.real_B_RL
        else:
            Fake_B = self.fake_B
            Real_B = self.real_B
            
        self.loss_G_L1 = self.criterionL1(Fake_B, Real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        
        # print("-------")
        # print("fake_B", Fake_B)
        # print("real_B", Real_B)
        # print("loss_G_GAN", self.loss_G_GAN)
        # print("loss_G_L1", self.loss_G_L1)
        # print("loss_G", self.loss_G)
        # print("-------")
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.input_RL_model()            # RL(G(A)) and RL(B)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
