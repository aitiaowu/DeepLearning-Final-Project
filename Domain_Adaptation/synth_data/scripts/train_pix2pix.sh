set -ex

# worked classfication
# python train.py --dataroot /content/datasets/depth2edge_real/ --name depth2edge_real --model our --netG unet_256 --netD basic --direction AtoB --lambda_entropy 30 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 3 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0
# python train.py --dataroot /content/datasets/depth2final_synth/ --dataroot_target /content/datasets/depth2final_real/ --name DA --model our --netG unet_256 --netD LeNet --direction AtoB --lambda_entropy 100 --lambda_DA 1 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 6 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 10 --n_epochs_decay 10 --continue_train --epoch 60 #--load_size 512 --crop_size 512 --preprocess none
#--load_size 341  --lr 0.0001 --continue_train --epoch 253 --n_epochs 5 --n_epochs_decay 30 


python train.py --dataroot /content/datasets/depth2final_synth/ --dataroot_target /content/datasets/depth2final_real/ --name DA --model our --netG unet_256 --netD LeNet --direction AtoB --lambda_entropy 100 --lambda_DA 10 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 6 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 10 --n_epochs_decay 10 --continue_train --epoch 60 #--load_size 512 --crop_size 512 --preprocess none
