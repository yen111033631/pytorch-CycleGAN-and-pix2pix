import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
# import .read_DQN
from .read_DQN import set_up_agent, get_q_values
import util.util as util
import ipdb, math

def spread_index_into_spherical(index, theta_num=8, shell_unit_length="exp", step_first=0.01, step_second=0.05):
    def special_exp(x, x1=0.01, x2=0.05):
        a = (x2 - x1) / (math.exp(1) - 1)
        b = 1
        c = x1 - a
        return a * math.exp(x-b) + c
    
    theta_unit_length = 360 / theta_num
    
    phi_num = theta_num / 2 - 1
    point_num = theta_num * phi_num + 2

    i = index
    r = i // point_num + 1
    index = i % point_num
    p = index % theta_num + 1
    t = index // theta_num + 1
    
    if index >= theta_num * phi_num:
        p = 0
        t = 0 if index == theta_num * phi_num else phi_num + 1
    
    p *= theta_unit_length
    t *= theta_unit_length
    if shell_unit_length == "exp":
        r = special_exp(r, step_first, step_second)
        
    elif isinstance(shell_unit_length, float):
        r *= shell_unit_length
    
    return r, t, p

# 定義球座標系轉換函數
def spherical_to_cartesian(radius, theta, phi):
    t2r = np.pi / 180
    theta *= t2r
    phi *= t2r
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    dis = np.asarray([x, y, z])
    return dis



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
        
        self.netG_loss_element = opt.netG_loss_setting.split("+")
        
        self.loss_names = self.netG_loss_element.copy()
        
        self.netD_existed = opt.netD_existed
        self.netD_setting = opt.netD
        if self.netD_existed:
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')
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
        self.agent = set_up_agent(opt.crop_size, opt.which_DQN)
        # opt.netD = "numerical" if self.is_added_DQN else opt.netD
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
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                          n_actions=self.agent.action_space.n)
            
            print("action space", self.agent.action_space.n)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        else:
            self.losses_list = []
            self.same_list = []
        
        # # ------------------
        # print(self.netD)
        # # ------------------
        self.image_tensor_size = torch.Size([opt.input_nc, opt.crop_size, opt.crop_size])

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
        
        # # ========================================
        # print(self.real_A.shape)
        # print(self.real_A[0, 0, 56, :10])
        # # ========================================

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        
        self.input_RL_model()  # RL(G(A)) and RL(B)

    def compute_loss(self):
        if self.is_added_DQN:            
            fake_B_tensor = self.fake_B_RL.clone().detach()
            real_B_tensor = self.real_B_RL.clone().detach()
        else:
        # 將self.fake_B和self.real_B轉換為PyTorch tensor類型
            fake_B_tensor = self.fake_B.clone().detach()
            real_B_tensor = self.real_B.clone().detach()
        
        # 創建L1損失函數的實例並計算損失
        l1_loss = torch.nn.L1Loss()
        
        # --
        cos_sim = F.cosine_similarity(fake_B_tensor, real_B_tensor, dim=-1)
        loss = (1 - cos_sim.mean()) * 500
        
        
        
        loss_l1 = l1_loss(fake_B_tensor, real_B_tensor)
        # --
        # print(round(loss.item(), 2), round(loss_l1.item(), 2))
        # print("-")
        
        
        
        real_B_action = real_B_tensor.argmax(1)[0].item()
        fake_B_action = fake_B_tensor.argmax(1)[0].item()
        is_same = real_B_action == fake_B_action
        # print("real_B_tensor", np.round(real_B_tensor.cpu().numpy(), 2))
        # print("fake_B_tensor", np.round(fake_B_tensor.cpu().numpy(), 2))
        # print(real_B_action, fake_B_action, is_same, round(loss.item(), 2), round(loss_l1.item(), 2))
        # print("---")
        # # print(loss)
        
        self.losses_list.append(loss.item())
        self.same_list.append(is_same)
        
    def input_RL_model(self):
        # ----------------------------------------------------------------------------
        # add DQN
        if self.is_added_DQN:
            # print(self.is_added_DQN)

            self.fake_B_RL = self.fake_B / 2.0 + 0.5
            self.fake_B_RL = self.agent.DQN(self.fake_B_RL)
            self.fake_B_RL_nor = (self.fake_B_RL + 250) / 250 * 100
            # self.fake_B_RL = self.fake_B_RL / 100

            self.real_B_RL = self.real_B / 2.0 + 0.5
            self.real_B_RL = self.agent.DQN(self.real_B_RL)
            self.real_B_RL_nor = (self.real_B_RL + 250) / 250 * 100
            # self.real_B_RL = self.real_B_RL / 100
        # ----------------------------------------------------------------------------
        

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # ----------------------------------------------------------------------------
        # Fake; stop backprop to the generator by detaching fake_B
        # if self.netD_input == "AB":
        #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # elif self.netD_input == "B":
        #     fake_AB = self.fake_B  if not(self.is_added_DQN)  else self.fake_B_RL  # we use conditional GANs; we need to feed output to the discriminator
        
        if self.netD_input == "AB" and not(self.is_added_DQN):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            action = None
        elif self.netD_input == "B" and not(self.is_added_DQN):
            fake_AB = self.fake_B
            action = None
        elif self.netD_input == "B" and self.is_added_DQN and self.netD_setting == "numerical":
            fake_AB = self.fake_B_RL
            action = None
        elif self.netD_input == "B" and self.is_added_DQN and self.netD_setting == "pixelnumerical":
            fake_AB = self.fake_B
            action = self.fake_B_RL
        else:
            print("netD set wrong")
            print("self.netD_input", self.netD_input)
            print("self.is_added_DQN", self.is_added_DQN)
            print("self.netD_setting", self.netD_setting)
            
            
        if action == None:
            pred_fake = self.netD(fake_AB.detach())
        else:
            pred_fake = self.netD(fake_AB.detach(), action.detach())
            
        # print("pred_fake netD output:", pred_fake.shape)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # ----------------------------------------------------------------------------
        # Real
        # if self.netD_input == "AB":
        #     real_AB = torch.cat((self.real_A, self.real_B), 1)
        # elif self.netD_input == "B":
        #     real_AB = self.real_B if not(self.is_added_DQN)  else self.real_B_RL     

        if self.netD_input == "AB" and not(self.is_added_DQN):
            real_AB = torch.cat((self.real_A, self.real_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            action = None
        elif self.netD_input == "B" and not(self.is_added_DQN):
            real_AB = self.real_B
            action = None
        elif self.netD_input == "B" and self.is_added_DQN and self.netD_setting == "numerical":
            real_AB = self.real_B_RL
            action = None
        elif self.netD_input == "B" and self.is_added_DQN and self.netD_setting == "pixelnumerical":
            real_AB = self.real_B
            action = self.real_B_RL
        else:
            print("netD set wrong")
        
        if action == None:
            pred_real = self.netD(real_AB.detach())
        else:
            pred_real = self.netD(real_AB.detach(), action.detach())
        
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        # ----------------------------------------------------------------------------

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # ----------------------------------------------------------------------------
        # First, G(A) should fake the discriminator
        if self.netD_existed:
            # if self.netD_input == "AB":
            #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            # elif self.netD_input == "B":
            #     fake_AB = self.fake_B if not(self.is_added_DQN)  else self.fake_B_RL 
                
            if self.netD_input == "AB" and not(self.is_added_DQN):
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
                action = None
            elif self.netD_input == "B" and not(self.is_added_DQN):
                fake_AB = self.fake_B
                action = None
            elif self.netD_input == "B" and self.is_added_DQN and self.netD_setting == "numerical":
                fake_AB = self.fake_B_RL
                action = None
            elif self.netD_input == "B" and self.is_added_DQN and self.netD_setting == "pixelnumerical":
                fake_AB = self.fake_B
                action = self.fake_B_RL
            else:
                print("netD set wrong")
                
            if action == None:
                pred_fake = self.netD(fake_AB)
            else:
                pred_fake = self.netD(fake_AB, action)  

            # pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # ----------------------------------------------------------------------------
        # Second, G(A) ~= B
        if self.is_added_DQN:
            self.loss_G_L1_RL = self.criterionL1(self.fake_B_RL, self.real_B_RL)
            self.loss_G_L1_RL_nor = self.criterionL1(self.fake_B_RL_nor, self.real_B_RL_nor)
            
            cos_sim = F.cosine_similarity(self.fake_B_RL, self.real_B_RL, dim=-1)            
            self.loss_G_cos_RL = (1 - cos_sim.mean()) * 500

            cos_sim_nor = F.cosine_similarity(self.fake_B_RL_nor, self.real_B_RL_nor, dim=-1)            
            self.loss_G_cos_RL_nor = (1 - cos_sim_nor.mean()) * 500
            
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # ----------------------------------------------------------------------------
        # combine loss and calculate gradients             
        losss = []
        for element in self.netG_loss_element:
            losss.append(getattr(self, 'loss_' + element))
        self.loss_G = sum(losss)
        
        self.loss_G.backward()
        # ----------------------------------------------------------------------------

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # self.input_RL_model()            # RL(G(A)) and RL(B)
        # ----------------------------------------------------------------------------
        # update D
        if self.netD_existed:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        # ----------------------------------------------------------------------------
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
        # ----------------------------------------------------------------------------
    
    def S2R_displacement(self, image_tensor: torch.Tensor):
        assert isinstance(image_tensor, torch.Tensor), "image_tensor should be torch.Tensor"
        assert image_tensor.shape[1:] == self.image_tensor_size, "image_tensor should be {}".format(self.image_tensor_size)
        
        with torch.no_grad():
            fake_B = self.netG(image_tensor)
            fake_B_RL = fake_B / 2.0 + 0.5
            fake_B_RL = self.agent.DQN(fake_B_RL) 
        
        action = fake_B_RL.argmax(1)[0].item()
        r, t, p = spread_index_into_spherical(action, 
                                              theta_num=8, 
                                              shell_unit_length=0.025)
        displacement = spherical_to_cartesian(r, t, p)


        
        # return displacement
        return fake_B, displacement
