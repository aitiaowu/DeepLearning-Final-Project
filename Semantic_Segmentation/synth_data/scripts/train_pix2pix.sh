set -ex

# Train Source model
python train.py --dataroot /content/datasets/depth2edge_synth/ --dataroot_target /content/datasets/depth2edge_synth/ --name depth2edge_synth --model our --netG encoder_decoder_256 --netD LeNet --direction AtoB --lambda_entropy 100 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 3 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 

# Train 
# python train.py --dataroot /content/datasets/depth2edge_real/ --dataroot_target /content/datasets/depth2edge_real/ --name depth2edge_synth --model our --netG encoder_decoder_256 --netD LeNet --direction AtoB --lambda_entropy 100 --dataset_mode onehotAligned --norm batch --pool_size 0 --input_nc 1 --output_nc 3 --batch_size 8 --checkpoints_dir /content/ --save_epoch_freq 1 --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 
