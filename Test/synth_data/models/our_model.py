import torch
from torch.autograd import Variable

from .base_model import BaseModel
from . import networks
import numpy as np
from collections import OrderedDict
import os
import math

class ourmodel(BaseModel):
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
            parser.add_argument('--lambda_entropy', type=float, default=100.0, help='weight for entropy loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_Entropy', 'D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B_visual', 'real_B_visual']
        #self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        self.netD = networks.define_D(256, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionEntropy = torch.nn.CrossEntropyLoss()
            self.criterionL1 = torch.nn.L1Loss()

            # self.IoU_weights = Variable(torch.tensor([1, 10, 1000])).to(self.device)
            self.IoU_weights = Variable(torch.tensor([0, 0, 0, 0, 0, 1])).to(self.device)
            # self.IoU_weights = Variable(torch.tensor([0.01, 0.1, 0.01, 0.1, 1, 1])).to(self.device)
            self.criterionIoU = networks.mIoULoss(weight=self.IoU_weights, n_classes=opt.output_nc)

            self.criterionDice = networks.diceLoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            optim_params = [
              {'params': self.netG.parameters(), 'lr': opt.lr},
              #{'params': self.netD.parameters(), 'lr': opt.lr}
              ]
            self.optimizer_G = torch.optim.Adam(optim_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input_train(self, input, target_input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        # source: real_A, real_B
        # target: target_real_A
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # data to train domain classifier
        self.target_real_A = target_input['A' if AtoB else 'B'].to(self.device)
        _ = target_input['B' if AtoB else 'A'].to(self.device)
        self.target_image_paths = target_input['A_paths' if AtoB else 'B_paths']
        self.gt_depth_style = torch.ones(self.real_A.shape[0], device=self.device).long()

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
        self.gt_depth_style = torch.zeros(self.real_A.shape[0], device=self.device).long()



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.source_features = self.netG(self.real_A)  # G(A)

        #self.fake_B = torch.softmax(self.fake_B, dim=1)
    
    def forward_D(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        _, self.source_features = self.netG(self.real_A)  # G(A)

        _, self.target_features = self.netG(self.target_real_A)  # G(A)

        self.discriminator_x = torch.cat([self.source_features, self.target_features])
        self.discriminator_y = torch.cat([torch.zeros(self.real_A.shape[0], device=self.device).long(),
        torch.ones(self.target_real_A.shape[0], device=self.device).long()])



    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        
        pred_depth_style = self.netD(self.discriminator_x.detach())

        # combine loss and calculate gradients
        self.loss_D = self.criterionEntropy(pred_depth_style, self.discriminator_y)

        self.loss_D.backward()

    def forward_G(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.source_features = self.netG(self.real_A)  # G(A)

        

    def backward_G(self):
        """Calculate GAN and L_CE loss for the generator"""
        # First, G(A) should fake the discriminator

        self.loss_D = torch.tensor(0)
        self.loss_G_GAN = torch.tensor(0)
        # pred_depth_style = self.netD(self.source_features)
        # target_pred_depth_style = self.netD(self.target_features)
        # self.discriminator_y = torch.ones(self.real_A.shape[0], device=self.device).long()
        # self.target_discriminator_y = torch.zeros(self.target_real_A.shape[0], device=self.device).long()
        # self.loss_G_GAN = self.criterionEntropy(pred_depth_style, self.discriminator_y) + self.criterionEntropy(target_pred_depth_style, self.target_discriminator_y)
        
        self.loss_G_Entropy = self.criterionIoU(self.fake_B, self.real_B) * self.opt.lambda_entropy

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Entropy #+ self.loss_G_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        # self.forward_D()                   # compute fake images: G(A)
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights

        self.forward_G()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.visualize()

    def visualize(self):
        # background: 0
        # body: 2
        # edge: 1



        labels = ['background',	'body', 'edge']
        cloth_map = np.array([[0,0,0],[0,0,200],[0,200,0]])


        self.fake_B_lable = torch.argmax(self.fake_B, dim=1)
        

        img = self.fake_B_lable.cpu()
        img = img[0]
        self.fake_B_visual = cloth_map[img]

        self.real_B_lable = torch.argmax(self.real_B, dim=1)

        img = self.real_B_lable.cpu()
        img = img[0]
        self.real_B_visual = cloth_map[img]

        self.real_A = self.real_A 

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        self.style_correct_num = 0
        self.loss_IoU_background = 0
        self.loss_IoU_1 = 0
        self.loss_IoU_2 = 0
        self.loss_IoU_3 = 0
        self.loss_IoU_4 = 0
        self.loss_IoU_5 = 0
        self.loss_IoU_mean = 0

        self.loss_gp_distance = 0
        self.loss_mean_body_distance = 0

        self.loss_gp_distance_median = 9999
        self.loss_mean_body_distance_median = 9999

        with torch.no_grad():
            self.forward()
            pred_depth_style = self.netD(self.source_features)
            pred_depth_style = torch.argmax(pred_depth_style, 1, keepdim=False).detach().cpu().numpy()
            gt_depth_style = self.gt_depth_style.cpu().numpy()
            if pred_depth_style == gt_depth_style:
              self.style_correct_num += 1
            


            self.compute_visuals()
            self.visualize()

            self.val_loss_names = ['IoU_background',	'IoU_1', 'IoU_2']
            # evaluation metric (IoU)
            #self.IoU = networks.TestIoULoss(n_classes=self.opt.output_nc)
            #self.loss_IoU = self.IoU(self.fake_B, torch.argmax(self.real_B, 1, keepdim=False).long())
            
            
            Pred = torch.argmax(self.fake_B, 1, keepdim=False).detach().cpu().numpy()
            GT = torch.argmax(self.real_B, 1, keepdim=False).cpu().numpy()
            

            
            
                    
            
            class_IoU, class_weight = GetIOU(Pred,GT, NumClasses=3, ClassNames=["background", "1", "2"], DisplyResults=False)

            self.loss_IoU_background += class_IoU[0]
            self.loss_IoU_1 += class_IoU[1]
            self.loss_IoU_2 += class_IoU[2]
            

    


        
        

    def validate(self, val_dataset):
        # Load validation dataset

        # from data import create_dataset
        # self.opt.phase = "val"
        # val_dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
        val_dataset_size = len(val_dataset)    # get the number of images in the dataset.
        # print('The number of validation images = %d' % val_dataset_size)
        self.loss_G_GAN = 0
        self.loss_G_Entropy = 0
        self.loss_D_fake = 0
        self.loss_D_real = 0

        self.loss_IoU_background = 0
        self.loss_IoU_1 = 0
        self.loss_IoU_2 = 0
        self.loss_IoU_3 = 0
        self.loss_IoU_4 = 0
        self.loss_IoU_5 = 0
        self.loss_IoU_mean = 0

        val_num = 0
        depth_style_correct_num = 0
        
        with torch.no_grad():
            for i, data in enumerate(val_dataset):
                self.set_input(data)
                self.forward()

                pred_depth_style = self.netD(self.source_features)
                pred_depth_style = torch.argmax(pred_depth_style, 1, keepdim=False).detach().cpu().numpy()
                gt_depth_style = self.gt_depth_style.cpu().numpy()
                if pred_depth_style == gt_depth_style:
                  depth_style_correct_num += 1
                val_num += 1

                

                # fake_AB = self.real_A * self.fake_B
                # pred_fake = self.netD(fake_AB)
                # real_AB = self.real_A * self.real_B
                # pred_real = self.netD(real_AB)
                # self.loss_G_GAN += self.criterionGAN(pred_fake, True)
                # self.loss_G_Entropy += self.criterionIoU(self.fake_B, self.real_B) * self.opt.lambda_entropy
                # self.loss_D_fake += self.criterionGAN(pred_fake, False)
                # self.loss_D_real += self.criterionGAN(pred_real, True)

                self.fake_B = torch.softmax(self.fake_B, dim=1)
                Pred = torch.argmax(self.fake_B, 1, keepdim=False).detach().cpu().numpy()
                GT = torch.argmax(self.real_B, 1, keepdim=False).cpu().numpy()
                class_IoU, class_weight = GetIOU(Pred, GT, NumClasses=6, ClassNames=["background", "1", "2", "3", "4", "5"], DisplyResults=False)
                self.loss_IoU_background += class_IoU[0]
                self.loss_IoU_1 += class_IoU[1]
                self.loss_IoU_2 += class_IoU[2]
                self.loss_IoU_3 += class_IoU[3]
                self.loss_IoU_4 += class_IoU[4]
                self.loss_IoU_5 += class_IoU[5]
                self.loss_IoU_mean += np.mean(class_IoU)

            # losses = self.get_current_losses()
            # message = 'Validation losses '
            # for k, v in losses.items():
            #     message += '%s: %.3f ' % (k, v*self.opt.batch_size/val_dataset_size)

            # print(message)  # print the message
            # log_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'loss_log.txt')
            # with open(log_name, "a") as log_file:
            #     log_file.write('%s\n' % message)  # save the message

            style_acc = depth_style_correct_num/val_num
            print("depth style accuracy: ", style_acc)
            log_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'val_log.txt')
            with open(log_name, "a") as log_file:
                log_file.write('background: {}, inner edge: {}, outer edge: {}, main body: {}, shoulder: {}, grasping point: {} mean: {}, style_acc: {}\n'.format(self.loss_IoU_background/val_dataset_size,
                self.loss_IoU_1/val_dataset_size, self.loss_IoU_2/val_dataset_size, self.loss_IoU_3/val_dataset_size, self.loss_IoU_4/val_dataset_size, 
                self.loss_IoU_5/val_dataset_size, self.loss_IoU_mean/val_dataset_size, style_acc))  # save the message

    def get_val_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""

        #errors_ret = OrderedDict()
        errors = {}
        for name in self.val_loss_names:
            if isinstance(name, str):
                errors[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors


import numpy as np
def GetIOU(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False):
    #Given A ground true and predicted labels per pixel return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate class intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate class Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union

    #------------Display results (optional)-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") IoU: "+str(ClassIOU[i]) + " class weight: " +str(ClassWeight[i]) )

       print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    #-------------------------------------------------------------------------------------------------

    return ClassIOU, ClassWeight
