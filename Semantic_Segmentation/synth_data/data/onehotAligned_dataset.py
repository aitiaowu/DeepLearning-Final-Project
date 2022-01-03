import os
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_segmentation
from data.numpy_folder import make_dataset
from PIL import Image
import torch
import random
import numpy as np
import cv2

class onehotAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # depth_style = int(AB_path[-5]) # depth style: real=1, synthetic=0
        # AB = cv2.imread(AB_path, cv2.IMREAD_UNCHANGED)
        AB = np.load(AB_path)
        height = AB.shape[0] # 512
        # split AB image into A and B
        A = np.float32(AB[:, :height])
        B = np.float32(AB[:, height:])

        # A = np.float32(AB)
        # B = np.float32(AB)
        # from matplotlib import pyplot as plt
        # plt.imshow(B)
        # plt.show()


        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.shape)

        
        deg = 45*random.uniform(0,1)
        A_transform, B_transform = get_transform_segmentation(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST, rotation=deg)
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to tranfsorms
        torch.manual_seed(seed)
        A = A_transform(Image.fromarray(A, 'F'))
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        B = B_transform(Image.fromarray(B, 'F'))

        # print("mean: ", np.mean(A[0].numpy()))
        # print("var: ", np.var(A[0].numpy()))
        # a
        # B = B[0].numpy()
        # from matplotlib import pyplot as plt
        # plt.imshow(B)
        # plt.show()

        B_np = B.numpy()


        # import numpy as np
        label_num = 6   # change the label num
        h = self.opt.crop_size   # change the label height
        w = self.opt.crop_size   # change the label width
        target_onehot = torch.zeros((label_num, h, w))
        for c in range(label_num):
            target_onehot[c][B_np == c] = 1
        target = torch.argmax(target_onehot, dim=0, keepdim=True)

        # a = target[0].numpy()
        # from matplotlib import pyplot as plt
        # plt.imshow(a)
        # # plt.show()
        # plt.savefig('/content/seg/seg_{}.png'.format(deg))
        
        # if depth_style == 0:
        #   depth_style_onehot = np.array([1,0]) # synthetic style
        # else:
        #   depth_style_onehot = np.array([0,1]) # real style

        # if depth_style == 0:
        #   depth_style = torch.tensor(np.array(0)) # synthetic style
        # else:
        #   depth_style = torch.tensor(np.array(1)) # real style
          

        return {'A': A, 'B': target_onehot, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
