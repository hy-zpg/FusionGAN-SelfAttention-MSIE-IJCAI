import utils.interpolations as interpolations
import numpy as np
import tqdm
from utils.storage import save_statistics, build_experiment_folder
from tensorflow.contrib import slim
from dagan_networks_wgan_with_matchingclassifier import *
from utils.sampling_with_matchingclassifier import sample_generator, sample_two_dimensions_generator, \
    sample_generator_store_into_file_for_classifier, sample_generator_store_into_file_for_quality, sample_generator_store_into_file
import time
import scipy.misc


def isNaN(num):
    return num != num


class ExperimentBuilder(object):
    def __init__(self, args, data):
        tf.reset_default_graph()
        self.continue_from_epoch = args.continue_from_epoch
        self.experiment_name = args.experiment_title
        self.saved_models_filepath, self.log_path, self.save_image_path = build_experiment_folder(self.experiment_name)
        self.num_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        # self.support_number = args.support_number
        self.selected_classes = args.selected_classes
        gen_depth_per_layer = args.generator_inner_layers
        discr_depth_per_layer = args.discriminator_inner_layers
        self.z_dim = args.z_dim
        self.num_generations = args.num_generations
        self.dropout_rate_value = args.dropout_rate_value
        self.data = data
        self.reverse_channels = False

        if args.generation_layers == 6:
            generator_layers = [64, 64, 128, 128, 256, 256]
            gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer,
                                gen_depth_per_layer, gen_depth_per_layer]
            generator_layer_padding = ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME"]
        else:
            generator_layers = [64, 64, 128, 128]
            gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                                gen_depth_per_layer, gen_depth_per_layer]
            generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

        discriminator_layers = [64, 64, 128, 128]
        discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                              discr_depth_per_layer]

        image_height = data.image_height
        image_width = data.image_width
        image_channel = data.image_channel

        self.classes = tf.placeholder(tf.int32)
        self.selected_classes = tf.placeholder(tf.int32)
        self.support_number = tf.placeholder(tf.int32)

        #### [self.input_x_i, self.input_y_i, self.input_global_y_i] --> [images, few shot label, global label]
        ## batch: [self.input_x_i, self.input_y_i, self.input_global_y_i]
        ## support: self.input_x_j, self.input_y_j, self.input_global_y_j]
        ## the input of discriminator: [self.input_x_j_selected, self.input_global_y_j_selected]
        self.input_x_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channel], 'batch')
        self.input_y_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.selected_classes],
                                        'y_inputs_bacth')
        self.input_global_y_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.training_classes],
                                               'y_inputs_bacth_global')

        self.input_x_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size,
                                                     self.data.selected_classes * self.data.support_number,
                                                     image_height, image_width,
                                                     image_channel], 'support')
        self.input_y_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size,
                                                     self.data.selected_classes * self.data.support_number,
                                                     self.data.selected_classes], 'y_inputs_support')
        self.input_global_y_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size,
                                                            self.data.selected_classes * self.data.support_number,
                                                            self.data.training_classes], 'y_inputs_support_global')

        self.input_x_j_selected = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                              image_channel], 'support_discriminator')
        self.input_global_y_j_selected = tf.placeholder(tf.float32,
                                                        [self.num_gpus, self.batch_size, self.data.training_classes],
                                                        'y_inputs_support_discriminator')

        # self.z_input = tf.placeholder(tf.float32, [self.batch_size*self.data.selected_classes, self.z_dim], 'z-input')
        # self.z_input_2 = tf.placeholder(tf.float32, [self.batch_size*self.data.selected_classes, self.z_dim], 'z-input_2')

        self.z_input = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input')
        self.z_input_2 = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input_2')

        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.z1z2_training = tf.placeholder(tf.bool, name='z1z2_training-flag')
        self.random_rotate = tf.placeholder(tf.bool, name='rotation-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')
        self.is_z2 = args.is_z2
        self.is_z2_vae = args.is_z2_vae

        self.matching = args.matching
        self.fce = args.fce
        self.full_context_unroll_k = args.full_context_unroll_k
        self.average_per_class_embeddings = args.average_per_class_embeddings
        self.restore_path = args.restore_path

        self.is_z2 = args.is_z2
        self.is_z2_vae = args.is_z2_vae
        self.loss_G = args.loss_G
        self.loss_D = args.loss_D
        self.loss_CLA = args.loss_CLA
        self.loss_FSL = args.loss_FSL
        self.loss_KL = args.loss_KL
        self.loss_recons_B = args.loss_recons_B
        self.loss_matching_G = args.loss_matching_G
        self.loss_matching_D = args.loss_matching_D
        self.loss_sim = args.loss_sim
        self.strategy = args.strategy

        #### training/validation/testin

        time_1 = time.time()
        dagan = DAGAN(batch_size=self.batch_size, input_x_i=self.input_x_i, input_x_j=self.input_x_j,
                      input_y_i=self.input_y_i, input_y_j=self.input_y_j, input_global_y_i=self.input_global_y_i,
                      input_global_y_j=self.input_global_y_j,
                      input_x_j_selected=self.input_x_j_selected,
                      input_global_y_j_selected=self.input_global_y_j_selected, \
                      selected_classes=self.data.selected_classes, support_num=self.data.support_number,
                      classes=self.data.training_classes,
                      dropout_rate=self.dropout_rate, generator_layer_sizes=generator_layers,
                      generator_layer_padding=generator_layer_padding, num_channels=data.image_channel,
                      is_training=self.training_phase, augment=self.random_rotate,
                      discriminator_layer_sizes=discriminator_layers,
                      discr_inner_conv=discr_inner_layers, is_z2=self.is_z2, is_z2_vae=self.is_z2_vae,
                      gen_inner_conv=gen_inner_layers, num_gpus=self.num_gpus, z_dim=self.z_dim, z_inputs=self.z_input,
                      z_inputs_2=self.z_input_2,
                      use_wide_connections=args.use_wide_connections, fce=self.fce, matching=self.matching,
                      full_context_unroll_k=self.full_context_unroll_k,
                      average_per_class_embeddings=self.average_per_class_embeddings,
                      loss_G=self.loss_G, loss_D=self.loss_D, loss_KL=self.loss_KL, loss_recons_B=self.loss_recons_B,
                      loss_matching_G=self.loss_matching_G, loss_matching_D=self.loss_matching_D,
                      loss_CLA=self.loss_CLA, loss_FSL=self.loss_FSL, loss_sim=self.loss_sim,
                      z1z2_training=self.z1z2_training)

        self.same_images = dagan.sample_same_images()

        # self.summary, self.losses, self.accuracy, self.graph_ops = classifier.init_train()

        self.total_train_batches = int(data.training_data_size / (self.batch_size * self.num_gpus))
        self.total_val_batches = int(data.validation_data_size / (self.batch_size * self.num_gpus))
        self.total_test_batches = int(data.testing_data_size / (self.batch_size * self.num_gpus))
        self.total_gen_batches = int(data.testing_data_size / (self.batch_size * self.num_gpus))
        self.init = tf.global_variables_initializer()

        time_2 = time.time()
        # print('time for constructing graph:',time_2 - time_1)

        self.tensorboard_update_interval = int(self.total_train_batches / 1 / self.num_gpus)
        self.total_epochs = 800
        self.is_generation_for_classifier = args.is_generation_for_classifier
        self.is_all_test_categories = args.is_all_test_categories

        # if self.continue_from_epoch == -1:
        #     save_statistics(self.log_path, ['epoch', 'total_d_train_loss_mean', 'total_d_val_loss_mean',
        #                                     'total_d_train_loss_std', 'total_d_val_loss_std',
        #                                     'total_g_train_loss_mean', 'total_g_val_loss_mean',
        #                                     'total_g_train_loss_std', 'total_g_val_loss_std'], create=True)

    def run_experiment(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # with tf.Session() as sess:
            time_4 = time.time()
            sess.run(self.init)
            print('time for initializing global parameters:', time.time() - time_4)

            # self.train_writer = tf.summary.FileWriter("{}/train_logs/".format(self.log_path),
            #                                           graph=tf.get_default_graph())
            # self.validation_writer = tf.summary.FileWriter("{}/validation_logs/".format(self.log_path),
            #                                                graph=tf.get_default_graph())

            log_name = "z2vae{}_z2{}_g{}_d{}_kl{}_cla{}_fzl{}_reconsB{}_matchingG{}_matchingD{}_sim{}_Net_batchsize{}_zdim{}".format(
                self.is_z2_vae, self.is_z2, self.loss_G, self.loss_D, self.loss_KL, self.loss_CLA, self.loss_FSL,
                self.loss_recons_B, self.loss_matching_G, self.loss_matching_D, self.loss_sim, self.batch_size,
                self.z_dim)

            self.train_writer = tf.summary.FileWriter("{}/train_logs/{}".format(self.log_path, log_name),
                                                      graph=sess.graph)
            self.validation_writer = tf.summary.FileWriter("{}/validation_logs/{}".format(self.log_path, log_name),
                                                           graph=sess.graph)

            self.train_saver = tf.train.Saver()
            self.val_saver = tf.train.Saver()

            # variable_names = [v.name for v in tf.trainable_variables()]
            # print(variable_names)

            start_from_epoch = 0
            if self.continue_from_epoch != -1:
                start_from_epoch = self.continue_from_epoch
                # checkpoint = "{}train_saved_model_{}_{}.ckpt".format(self.saved_models_filepath, self.experiment_name, self.continue_from_epoch)
                checkpoint = self.restore_path
                variables_to_restore = []
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    # print(var)
                    variables_to_restore.append(var)

                tf.logging.info('Fine-tuning from %s' % checkpoint)

                fine_tune = slim.assign_from_checkpoint_fn(
                    checkpoint,
                    variables_to_restore,
                    ignore_missing_vars=True)
                fine_tune(sess)

            self.iter_done = 0
            self.disc_iter = 1
            self.gen_iter = 1
            best_d_val_loss = np.inf

            ### z preprocess
            self.spherical_interpolation = False
            if self.spherical_interpolation:
                self.z_vectors = interpolations.create_mine_grid(rows=1, cols=self.num_generations, dim=self.z_dim,
                                                                 space=3, anchors=None, spherical=True, gaussian=True)
                self.z_vectors_2 = interpolations.create_mine_grid(rows=1, cols=self.num_generations, dim=self.z_dim,
                                                                   space=3, anchors=None, spherical=True, gaussian=True)

                self.z_2d_vectors = interpolations.create_mine_grid(rows=self.num_generations,
                                                                    cols=self.num_generations,
                                                                    dim=100, space=3, anchors=None,
                                                                    spherical=True, gaussian=True)
                self.z_2d_vectors_2 = interpolations.create_mine_grid(rows=self.num_generations,
                                                                      cols=self.num_generations,
                                                                      dim=100, space=3, anchors=None,
                                                                      spherical=True, gaussian=True)


            else:
                self.z_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                self.z_vectors_2 = np.random.normal(size=(self.num_generations, self.z_dim))

                self.z_2d_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                self.z_2d_vectors_2 = np.random.normal(size=(self.num_generations, self.z_dim))

            self.support_vector = [x for x in range(10)]

            ### training with train set with parameter update, validation without training, epoch
            if self.is_all_test_categories > 0:
                sampler = sample_generator_store_into_file
            else:
                sampler = sample_generator_store_into_file_for_classifier
            image_name = "z2vae{}_z2{}_g{}_d{}_kl{}_cla{}_fzl{}_reconsB{}_matchingG{}_matchingD{}_sim{}_Net_batchsize{}_zdim{}".format(
                self.is_z2_vae, self.is_z2, self.loss_G, self.loss_D, self.loss_KL, self.loss_CLA, self.loss_FSL,
                self.loss_recons_B, self.loss_matching_G, self.loss_matching_D, self.loss_sim, self.batch_size,
                self.z_dim)

            if self.continue_from_epoch >= 0:
                # print('starting sampling')
                similarities_list = []
                f_encode_z_list = []
                matching_feature_list = []
                if (self.total_gen_batches < 10):
                    self.total_gen_batches = 10
                with tqdm.tqdm(total=self.total_gen_batches) as pbar_samp:
                    for i in range(self.total_gen_batches):
                        x_test_i_selected_classes, x_test_j, y_test_i_selected_classes, y_test_j, y_global_test_i_selected_classes, y_global_test_j = self.data.get_test_batch()
                        # np.random.seed(i)
                        # self.z_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                        # self.z_vectors_2 = np.random.normal(size=(self.num_generations, self.z_dim))
                        if i >= 0:
                            for j in range(1):
                                before_sample = time.time()
                                x_test_i = x_test_i_selected_classes[:, :, j, :, :, :]
                                y_test_i = y_test_i_selected_classes[:, :, j, :]
                                y_global_test_i = y_global_test_i_selected_classes[:, :, j, :]

                                support_index = int(np.random.choice(self.data.support_number, size=1))
                                x_test_j_selected = x_test_j[:, :, support_index, :, :, :]
                                y_test_j_selected = y_test_j[:, :, support_index, :]
                                y_global_test_j_selected = y_global_test_j[:, :, support_index, :]

                                _, _, _ = sampler(num_generations=self.num_generations,
                                                                           sess=sess,
                                                                           same_images=self.same_images,
                                                                           input_a=self.input_x_i,
                                                                           input_b=self.input_x_j,
                                                                           input_y_i=self.input_y_i,
                                                                           input_y_j=self.input_y_j,
                                                                           input_global_y_i=self.input_global_y_i,
                                                                           input_global_y_j=self.input_global_y_j,
                                                                           classes=self.classes,
                                                                           classes_selected=self.selected_classes,
                                                                           number_support=self.support_number,
                                                                           z_input=self.z_input,
                                                                           z_input_2=self.z_input_2,
                                                                           # selected_global_x_j = self.input_x_j_selected,
                                                                           # selected_global_y_j=self.input_global_y_j_selected,

                                                                           # conditional_inputs=x_test_i,
                                                                           # y_input_i = y_test_i,
                                                                           conditional_inputs=x_test_j_selected,
                                                                           y_input_i=y_test_j_selected,

                                                                           support_input=x_test_j,
                                                                           y_input_j=y_test_j,
                                                                           y_global_input_i=y_global_test_i,
                                                                           y_global_input_j=y_global_test_j,
                                                                           classes_number=self.data.training_classes,
                                                                           selected_classes=self.data.selected_classes,
                                                                           support_number=self.data.support_number,
                                                                           # input_global_x_j_selected = x_test_j_selected,
                                                                           # input_global_y_j_selected = y_global_test_j_selected,
                                                                           z_vectors=self.z_vectors,
                                                                           z_vectors_2=self.z_vectors_2,
                                                                           data=self.data,
                                                                           batch_size=self.batch_size,
                                                                           file_name="{}/{}_{}_{}.png".format(
                                                                               self.save_image_path,
                                                                               image_name,
                                                                               self.continue_from_epoch,
                                                                               i),
                                                                           dropout_rate=self.dropout_rate,
                                                                           dropout_rate_value=self.dropout_rate_value,
                                                                           training_phase=self.training_phase,
                                                                           z1z2_training=self.z1z2_training,
                                                                           is_training=False,
                                                                           training_z1z2=False)
                                if self.is_all_test_categories > 0:
                                    ##### storing the test images into correspoding
                                    print('real',np.shape(x_test_i_selected_classes))


                                    file_name = "{}/{}_{}_{}.png".format(self.save_image_path,
                                                                               image_name,
                                                                               self.continue_from_epoch,i)
                                    for j in range(self.batch_size):
                                        current_path_classifier = file_name.split('//')[0] + '_forclassifier' + '/{}/'.format(np.argmax(y_global_test_i_selected_classes[:, j]))
                                        print('testing number',np.shape(x_test_i_selected_classes)[2])
                                        for k in range(np.shape(x_test_i_selected_classes)[2]):
                                            current_name = current_path_classifier + 'sample{}.png'.format(k)
                                            current_iamge = x_test_i_selected_classes[0,j,k]
                                            print('image shape',np.shape(current_iamge))
                                            print('current path',current_name)
                                            scipy.misc.imsave(current_name, current_iamge)