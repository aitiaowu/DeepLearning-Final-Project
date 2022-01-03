set -ex


# test result
# python test_phase1.py --dataroot /content/datasets/depth2edge_real --name depth2edge_synth_with_style --model our --netG unet_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 3 --epoch 349

# test result on finetune
# python test_phase1.py --dataroot /content/datasets/depth2edge_real --name depth2seg_synth --model our --netG unet_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 3 --epoch 300


# for EPOCH in {200..300}
# do
#     echo ${EPOCH}
#     python test_phase1.py --dataroot /content/datasets/depth2more_edges_synth/ --name depth2more_edges_synth --model our --netG unet_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 6 --epoch ${EPOCH} --num_test 100

# done
#--load_size 512 --crop_size 512 --num_test 100

python test_phase1.py --dataroot /content/datasets/depth2more_edges_synth/ --name depth2more_edges_synth --model our --netG unet_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 6 --epoch 300 
#--load_size 512 --crop_size 512


# test with domain adaption
#python test_phase1.py --dataroot /content/datasets/depth2edge_real/ --name DA --model our --netG unet_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 3 --epoch 179


# for EPOCH in {1..45}
# do
#     echo ${EPOCH}
#     python test_phase1.py --dataroot /content/datasets/depth2more_edges_real_v2/ --name DA --model our --netG unet_256 --netD LeNet --direction AtoB --dataset_mode onehotAligned --norm batch --checkpoints_dir /content/ --results_dir /content/result/ --input_nc 1 --output_nc 6 --epoch ${EPOCH} --num_test 100   
# done