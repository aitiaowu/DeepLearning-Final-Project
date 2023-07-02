# DeepLearning-Final-Project

## Demo
Click to see [demo](https://github.com/aitiaowu/DeepLearning-Final-Project/blob/main/test_Demo.ipynb) using provided [model](https://github.com/aitiaowu/DeepLearning-Final-Project/tree/main/ckpt).
## Preamble
This codebase was developed and tested with Python 3.6, Pytorch 1.4.0, CUDA 11.2. The framework is based on [pytorch-CycleGAN-and-pix2pix
](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Prerequisite
```bash
%cd Semantic_Segmentation/synth_data/
!pip3 install -r requirements.txt
```

## Training Semantic segmentation
You should be able to train the model by running the following command.
```bash
!python train.py --dataroot /content/datasets/depth2edge_synth/ --dataroot_target /content/datasets/depth2edge_synth/ --name depth2edge_synth --model our --netG encoder_decoder_256 --netD LeNet --direction AtoB --lambda_entropy 100 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 3 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 
```

## Training with Domain adaptation
After loading the model trained only Semantic segmentation,you should be able to train the model by running the following command.
```bash
!python train.py --dataroot /content/datasets/depth2edge_synth/ --dataroot_target /content/datasets/depth2final_real_all_depth/ --name DA --model our --netG encoder_decoder_256 --netD LeNet --direction AtoB --lambda_entropy 20 --lambda_DA 100 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 3 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 30 --n_epochs_decay 30 --continue_train --epoch 176 #--load_size 512 --crop_size 512 --preprocess none
```

## Test
You should be able to test the model by running the following command.
```bash
!python test_phase1.py --epoch 50 --dataroot /content/datasets/depth2edge_real/ --name DA --model our --netG encoder_decoder_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 3 
```

Note: You can also see the whole Training/Testing process [here](https://github.com/aitiaowu/DeepLearning-Final-Project/blob/main/02456_Final_Project.ipynb).


