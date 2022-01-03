"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import loop_iterable
import warnings
warnings.filterwarnings("ignore")
import math

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # dataset: source dataset target_dataset: target dataset
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    target_opt = opt
    target_opt.dataroot = target_opt.dataroot_target
    target_dataset = create_dataset(target_opt)
    target_dataset_size = len(target_dataset)
    print("!!!!!!!!!!")
    print('The number of style training images = %d' % target_dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.model_names = ['G']
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Validation dataset
    val_opt = TrainOptions().parse()  # get test options
    val_opt.phase = "test"
    # hard-code some parameters for test
    val_opt.num_threads = 0   # test code only supports num_threads = 0
    val_opt.batch_size = 1    # test code only supports batch_size = 1
    val_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    val_opt.no_rotation = True
    val_dataset1 = create_dataset(val_opt)  # create a dataset given opt.dataset_mode and other options
    val_opt.dataroot = target_opt.dataroot # set validataion dataset on Real style depth
    val_dataset2 = create_dataset(val_opt)  # create a dataset given opt.dataset_mode and other options

    print('The number of validation images = %d' % len(val_dataset1))
        
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        batch_iterator = zip(loop_iterable(dataset), loop_iterable(target_dataset))
        #for i, data in enumerate(dataset):  # inner loop within one epoch


        for i in range(math.floor(target_dataset_size/(opt.batch_size))):
            data, target_data = next(batch_iterator)
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # model_front2back.set_input(data)
            # data["A_back"] = model_front2back.forward()

            model.set_input_train(data, target_data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(i)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            # model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        # Validation losses
        # model.eval()
        # model.validate(val_dataset1, 'synthetic')
        # model.validate(val_dataset2, 'real')
        # model.train_state()
        