set -ex

# Train Source model
python train.py --dataroot /content/datasets/depth2final_synth/ --dataroot_target /content/datasets/depth2final_synth/ --name depth2final_synth --model our --netG unet_256 --netD LeNet --direction AtoB --lambda_entropy 1 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 6 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 30 --n_epochs_decay 30 

# Train Target model
# python train.py --dataroot /content/datasets/depth2final_real/ --dataroot_target /content/datasets/depth2final_real/ --name depth2final_real --model our --netG unet_256 --netD LeNet --direction AtoB --lambda_entropy 1 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 6 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 150 --n_epochs_decay 150 

# Train Source ED model
# python train.py --dataroot /content/datasets/depth2final_synth/ --dataroot_target /content/datasets/depth2final_synth/ --name depth2final_synth_ed --model our --netG encoder_decoder_256 --netD LeNet --direction AtoB --lambda_entropy 1 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 6 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 150 --n_epochs_decay 150 
