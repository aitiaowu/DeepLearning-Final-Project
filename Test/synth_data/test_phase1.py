"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.no_rotation = True
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    opt.eval = True
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    IoU_background = 0
    IoU_1 = 0
    IoU_2 = 0
    IoU_3 = 0
    IoU_4 = 0
    IoU_5 = 0
    IoU_mean = 0
    counter = 0
    style_correct_num = 0
    gp_distance = 0
    mean_body_distance = 0
    all_gp_dist = []
    all_mean_dist = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        losses = model.get_val_losses()
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        IoU_background += losses['IoU_background']
        IoU_1 += losses['IoU_1']
        IoU_2 += losses['IoU_2']
        IoU_3 += losses['IoU_3']
        IoU_4 += losses['IoU_4']
        IoU_5 += losses['IoU_5']
        IoU_mean += losses['IoU_mean']
        style_correct_num += losses['style_correct_num']
        gp_distance += losses['gp_distance']
        mean_body_distance += losses['mean_body_distance']
        counter += 1

        if not (losses['gp_distance_median']==10086):
            all_gp_dist.append(losses['gp_distance_median'])
            
        
        if not (losses['mean_body_distance_median']==10086):
            all_mean_dist.append(losses['mean_body_distance_median'])

        style_acc = style_correct_num/counter  
    import statistics
    print("IoU background: {}, body: {}, top: {}, middle: {}, bottom: {}, grasping point: {} mean: {}, style_acc: {}, gp distance: {}, mean_body distance: {}, median gp: {}, median mean: {}".format(IoU_background/counter, IoU_1/counter, IoU_2/counter, IoU_3/counter, IoU_4/counter, IoU_5/counter, IoU_mean/counter, style_acc, statistics.mean(all_gp_dist), statistics.mean(all_mean_dist), statistics.median(all_gp_dist), statistics.median(all_mean_dist)))
    with open("/content/result.txt", "a") as file_object:
        file_object.write("IoU background: {}, body: {}, top: {}, middle: {}, bottom: {}, grasping point: {} mean: {}, style_acc: {}, gp distance: {}, mean_body distance: {}, median gp: {}, median mean: {}\n".format(IoU_background/counter, IoU_1/counter, IoU_2/counter, IoU_3/counter, IoU_4/counter, IoU_5/counter, IoU_mean/counter, style_acc, statistics.mean(all_gp_dist), statistics.mean(all_mean_dist), statistics.median(all_gp_dist), statistics.median(all_mean_dist)))
    print("gp", all_gp_dist)
    print("mean body", all_mean_dist)
    print(len(all_mean_dist))
    # print("gp", all_gp_dist)
    # print("body", all_mean_dist)
    webpage.save()  # save the HTML
