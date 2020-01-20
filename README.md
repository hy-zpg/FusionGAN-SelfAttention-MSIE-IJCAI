#####
Attention: from the last block of unet
Similarity: random uniform




######  dataset and experimental setting
For emnist and omniglot
experiment: layer_size   --generator_layers = [64, 64, 128, 128]
                         --gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                            gen_depth_per_layer, gen_depth_per_layer]
            image_size   --[28,28,1]

architecture 1conection: idx-2 > 0



For vggface
experiment: layer_size   --generator_layers = [64, 64, 128, 128]
                         --gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                            gen_depth_per_layer, gen_depth_per_layer]
            image_size   --[84,84,3]

architecture 1conection: idx-2 > 0



experiment: layer_size --generator_layers = [64, 64, 128, 128, 256, 256]
                       --gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                            gen_depth_per_layer, gen_depth_per_layer,gen_depth_per_layer, gen_depth_per_layer]
            image_size --[128,128,3]

architecture 1conection: idx-2 > 0



###### To continue training the matchingGAN

CUDA_VISIBLE_DEVICES=1 nohup python -u train_dagan_with_matchingclassifier.py --dataset flowers --generation_layers 6 --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title flowers1way3shot6layers3connectionSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path ./flowers1way3shot6layers3connectionSA/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize128_epoch430.ckpt --continue_from_epoch 430 > vaen2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u train_dagan_with_matchingclassifier.py --dataset animals --generation_layers 6 --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title animals1way3shot6layers3connectionSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path ./animals1way3shot6layers3connectionSA/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize128_epoch430.ckpt --continue_from_epoch 430 > vaen1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python train_dagan_with_matchingclassifier.py --dataset vggface --is_all_test_categories 1 --generation_layers 6 --image_width 64 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 512 --experiment_title Augmented_vggface_3shotSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path ./vggface1way3shot6layers3connectionNEW/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD0.0_sim1.0_Net_batchsize20_classencodedim64_imgsize64_epoch215.ckpt --continue_from_epoch 216




###########AUTOMATIC EVALUATION############

given the path of trained model, aiming at calculating the FID, IS, LPIPS and classifiers with augmented images

#### 1.Generating images for those unseen categories
   Two branch: 
      I(VISUALIZATION): selecting specific categories for visualization
      II(QUALITY && CLASSIFIER): generating samples for those unseen categories

0(VISUALIZATION):
	.. prepared selected categories: stored into file [DatasetVggfaceFlowersAnimalsTest]
	
	.. transfering it into npy: 
	CUDA_VISIBLE_DEVICES=1 python data_preparation.py --dataroot ./DatasetVggfaceFlowersAnimalsTest/animals  --storepath  TestVggfaceFlowersAnimalsnpy/dataset/ --image_width 128 --image_channel 3 --augmented_support 3

    CUDA_VISIBLE_DEVICES=1 python data_preparation.py --dataroot ./DatasetVggfaceFlowersAnimalsTest/flowers  --storepath  TestVggfaceFlowersAnimalsnpy/dataset/ --image_width 128 --image_channel 3 --augmented_support 3

    CUDA_VISIBLE_DEVICES=1 python data_preparation.py --dataroot ./DatasetVggfaceFlowersAnimalsTest/vggfaces  --storepath  TestVggfaceFlowersAnimalsnpy/dataset/ --image_width 128 --image_channel 3 --augmented_support 3

    .. generating 1shot in dagan setting
    CUDA_VISIBLE_DEVICES=0  python  test_dagan_with_matchingclassifier_for_generation.py --dataset animals --is_all_test_categories 0 --is_generation_for_classifier 1 --image_width 64 --generation_layers 4 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 4 --experiment_title VISUALIZATION_animals_1shotNEW --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 1 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 1 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 0 --loss_sim 1 --z_dim 64 --strategy 1 --restore_path ./animals1way1shot4layers2connection/saved_models/train_LOSS_z2vae0_z21_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize64_epoch3.ckpt --continue_from_epoch 106

    CUDA_VISIBLE_DEVICES=1  python  test_dagan_with_matchingclassifier_for_generation.py --dataset flowers --is_all_test_categories 0 --is_generation_for_classifier 1 --image_width 64 --generation_layers 4 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 4 --experiment_title VISUALIZATION_flowers_1shotNEW --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 1 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 1 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 0 --loss_sim 1 --z_dim 64 --strategy 1 --restore_path ./flowers1way1shot4layers2connection/saved_models/train_LOSS_z2vae0_z21_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize64_epoch5.ckpt --continue_from_epoch 111

    CUDA_VISIBLE_DEVICES=1  python  test_dagan_with_matchingclassifier_for_generation.py --dataset vggface --is_all_test_categories 0 --is_generation_for_classifier 1 --generation_layers 4 --image_width 64 --generation_layers 4 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 4 --experiment_title VISUALIZATION_vggfaces_1shotNEW --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 1 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 1 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 0 --loss_sim 1 --z_dim 64 --strategy 1 --restore_path ./vggface1way1shot4layers2connection/saved_models/train_LOSS_z2vae0_z21_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize64_epoch3.ckpt  --continue_from_epoch 81

    parameters: 
	--dataset flowers --is_all_test_categories 0 --generation_layers 4 --support_number 3  --experiment_title  VISUALIZATION_animals_3shotSA (fixed form: VISUALIZATION_DATSETS_SUPPORTNUMshotMETHOD)

	results:
	generated images for visualization: VISUALIZATION/animals/3shotSA_quality(each category include [num_generation] images), selecting samples for visualization
	

I(QUALITY && CLASSIFIER):
	
	CUDA_VISIBLE_DEVICES=0 nohup python -u test_dagan_with_matchingclassifier_for_generation.py --dataset animals --is_all_test_categories 1  --generation_layers 6 --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 512 --experiment_title Augmented_animals_3shotSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path  ./animals1way3shot6layers3connectionSA/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize128_epoch430.ckpt  --continue_from_epoch 430 > vaen1.log 2>&1 &

	CUDA_VISIBLE_DEVICES=1 nohup python -u test_dagan_with_matchingclassifier_for_generation.py --dataset flowers --is_all_test_categories 1  --generation_layers 6 --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 512 --experiment_title Augmented_flowers_3shotSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path  ./flowers1way3shot6layers3connectionSA/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize128_epoch430.ckpt  --continue_from_epoch 430 > vaen2.log 2>&1 &


	CUDA_VISIBLE_DEVICES=1 nohup python -u test_dagan_with_matchingclassifier_for_generation.py --dataset vggface --is_all_test_categories 0  --generation_layers 6 --image_width 64 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 48 --experiment_title Augmented_vggfaces_3shotniuli --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path  ./vggface1way3shot6layers3connectionNEW/saved_models/valid_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD0.0_sim1.0_Net_batchsize20_classencodedim64_imgsize64_epoch216.ckpt  --continue_from_epoch 217 > vaen2.log 2>&1 &


	parameters: 
	--dataset flowers --is_all_test_categories 1 --generation_layers 6 --support_number 3  --experiment_title  Augmented_animals_3shotSA (fixed form: Augmented_DATSETS_SUPPORTNUMshotMETHOD)

	results:
	generated images for quality: Augmented/animals/3shotSA_quality(each category include [num_generation] images)
	generated images for classifier: Augmented/animals/3shotSA_classifier(each category include [num_generation+support_num] images)
 

#### 2. transfering images stored in the folders into npy form
	

I(QUALITY):
	CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/flowers/3shotSA/visual_outputs_forquality --storepath  Augmentednpy/dataset_forquality/ --image_width 128 --image_channel 3 --augmented_support 100

	CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/animals/3shotSA/visual_outputs_forquality --storepath  Augmentednpy/dataset_forquality/ --image_width 128 --image_channel 3 --augmented_support 100

	parameters: --dataroot ./AugmentedQuality/animals/3shotSA/visual_outputs_forquality(consistent to AugmentedQuality_flowers_3shotSA)
	--storepath  Augmentednpy/dataset_forquality/(fixed, unified) --augmented_support 100(the number of generated images for each category)

	results: npy file flowers_100.npy(dataset_augmented_support) stored into Augmentednpy/dataset_forquality/


II(CLASSIFIER):
	CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/flowers/3shotSA/visual_outputs_forclassifier  --storepath  Augmentednpy/dataset_forclassifier/ --image_width 128 --image_channel 3 --augmented_support 545(512+3+30)

	CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/animals/3shotSA/visual_outputs_forclassifier  --storepath  Augmentednpy/dataset_forclassifier/ --image_width 128 --image_channel 3 --augmented_support 545(512+3)

	parameters: --dataroot ./AugmentedClassifier/animals/3shotSA/visual_outputs(consistent to AugmentedQuality_flowers_3shotSA)
	--storepath  Augmentednpy/dataset_forclassifier/(fixed, unified) --augmented_support 100(the number of generated images for each category + used support images)

	results: npy file flowers_512.npy(dataset_augmented_support) stored into Augmentednpy/dataset_forclassifier/





#### 3. calculating the FID/IS/LPIPS from the generated npy file stored into AugmentedQualitynpy/dataset/

   I. calculating FID/IS

   CUDA_VISIBLE_DEVICES=0 python cal-gan-metrics.py --dataset animals  --samples_each_category  50
   CUDA_VISIBLE_DEVICES=1 python cal-gan-metrics.py --dataset flowers  --samples_each_category  50

   II. calculating LPIPS
   cd LPIPS-pytorch
   CUDA_VISIBLE_DEVICES=0 nohup python -u compute_dists_pair.py -d ../AugmentedQuality/flowers/3shotSA/visual_outputs/0 -o imgs/example_dists_pair.txt --use_gpu > 1.log 2>&1 &

   CUDA_VISIBLE_DEVICES=0 nohup python -u compute_dists_pair.py -d ../AugmentedQuality/animals/3shotSA/visual_outputs/0 -o imgs/example_dists_pair.txt --use_gpu > 2.log 2>&1 &



#### 4. training classifer with generated images

	CUDA_VISIBLE_DEVICES=0 nohup python -u train_classifier_with_augmented_images.py --dataset animals --episodes_number 10 --selected_classes 5  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedClassifier_animals_augmented512  --image_width 128  --image_height 128 --image_channel 3  > 11.log 2>&1 & 


	CUDA_VISIBLE_DEVICES=1 nohup python -u train_classifier_with_augmented_images.py --dataset flowers --episodes_number 10 --selected_classes 5 --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedClassifier_flowers_augmented512  --image_width 128  --image_height 128 --image_channel 3  > 22.log 2>&1 & 



#############################
sequence running For vggface
#############################
Step 1: generating
CUDA_VISIBLE_DEVICES=0 python test_dagan_with_matchingclassifier_for_generation.py --dataset vggface --is_all_test_categories 1 --generation_layers 6 --image_width 64 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 512 --experiment_title Augmented_vggface_5shotSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 5 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path ./vggface1way3shot6layers3connectionNEW/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD0.0_sim1.0_Net_batchsize20_classencodedim64_imgsize64_epoch215.ckpt --continue_from_epoch 611


Step2: transfering images into npy (quality+classifier)
CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/vggface/5shotSA/visual_outputs_forquality --storepath  Augmentednpy/dataset_forquality/ --image_width 128 --image_channel 3 --augmented_support 100
CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/vggface/5shotSA/visual_outputs_forclassifier  --storepath  Augmentednpy/dataset_forclassifier/ --image_width 128 --image_channel 3 --augmented_support 548(512+5+30)

Step3: calculating FID and IS
CUDA_VISIBLE_DEVICES=1 python cal-gan-metrics.py --dataset vggface  --samples_each_category  50


Step4: calculating LPIPS
cd  LPIPS-pytorch
CUDA_VISIBLE_DEVICES=0 nohup python -u compute_dists_pair.py -d ../AugmentedQuality/vggface/5shotSA/visual_outputs/0 -o imgs/example_dists_pair.txt --use_gpu > 1.log 2>&1 &


Step5: trianing linear classifier with generated images and test on the remaining samples
CUDA_VISIBLE_DEVICES=1 python train_classifier_with_augmented_images.py --dataset vggface --episodes_number 10 --selected_classes 5  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedClassifier_vggface5way5shot_augmented512  --image_width 256  --image_height 256 --image_channel 3  

CUDA_VISIBLE_DEVICES=0 python train_classifier_with_augmented_images.py --dataset vggface --episodes_number 10 --selected_classes 10  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedClassifier_vggface10way5shot_augmented512  --image_width 256  --image_height 256 --image_channel 3 




#############################
sequence running For flowers
#############################
Step 1: generating
CUDA_VISIBLE_DEVICES=1 nohup python -u test_dagan_with_matchingclassifier_for_generation.py --dataset flowers --is_all_test_categories 1  --generation_layers 6 --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 512 --experiment_title Augmented_flowers_3shotSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path  ./flowers1way3shot6layers3connectionSA/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize128_epoch430.ckpt  --continue_from_epoch 430 > vaen2.log 2>&1 &


Step2: transfering images into npy (quality+classifier)
CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/flowers/3shotSA/visual_outputs_forquality --storepath  Augmentednpy/dataset_forquality/ --image_width 128 --image_channel 3 --augmented_support 100
CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/flowers/3shotSA/visual_outputs_forclassifier  --storepath  Augmentednpy/dataset_forclassifier/ --image_width 128 --image_channel 3 --augmented_support 545(512+3+30)


Step3: calculating FID and IS
CUDA_VISIBLE_DEVICES=1 python cal-gan-metrics.py --dataset flowers  --samples_each_category  50


Step4: calculating LPIPS
cd  LPIPS-pytorch
CUDA_VISIBLE_DEVICES=0 nohup python -u compute_dists_pair.py -d ../AugmentedQuality/flowers/3shotSA/visual_outputs/0 -o imgs/example_dists_pair.txt --use_gpu > 1.log 2>&1 &


Step5: trianing linear classifier with generated images and test on the remaining samples
CUDA_VISIBLE_DEVICES=1 nohup python -u train_classifier_with_augmented_images.py --dataset flowers --episodes_number 10 --selected_classes 5  --classification_total_epoch 50  --experiment_title AugmentedClassifier_flowers_augmented512  --image_width 128  --image_height 128 --image_channel 3  > 22.log 2>&1 & 



CUDA_VISIBLE_DEVICES=1 python train_classifier_with_augmented_images.py --dataset flowers --episodes_number 10 --selected_classes 5  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedClassifier_flowers_augmented512  --image_width 128  --image_height 128 --image_channel 3  
#############################
sequence running For flowers
#############################




#############################
sequence running For animals
#############################
Step 1: generating
CUDA_VISIBLE_DEVICES=0 nohup python -u test_dagan_with_matchingclassifier_for_generation.py --dataset animals --is_all_test_categories 1  --generation_layers 6 --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 512 --experiment_title Augmented_animals_3shotSA --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 64 --strategy 1 --restore_path  ./animals1way3shot6layers3connectionSA/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_classencodedim64_imgsize128_epoch430.ckpt  --continue_from_epoch 430 > vaen1.log 2>&1 &


Step2: transfering images into npy (quality+classifier)
CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/animals/3shotSA/visual_outputs_forquality --storepath  Augmentednpy/dataset_forquality/ --image_width 128 --image_channel 3 --augmented_support 100
CUDA_VISIBLE_DEVICES=0 python data_preparation.py --dataroot ./Augmented/animals/3shotSA/visual_outputs_forclassifier  --storepath  Augmentednpy/dataset_forclassifier/ --image_width 128 --image_channel 3 --augmented_support 545(512+3+30)


Step3: calculating FID and IS
CUDA_VISIBLE_DEVICES=1 python cal-gan-metrics.py --dataset animals  --samples_each_category  50


Step4: calculating LPIPS
cd  LPIPS-pytorch
CUDA_VISIBLE_DEVICES=0 nohup python -u compute_dists_pair.py -d ../AugmentedQuality/flowers/3shotSA/visual_outputs/0 -o imgs/example_dists_pair.txt --use_gpu > 1.log 2>&1 &


Step5: trianing linear classifier with generated images and test on the remaining samples
CUDA_VISIBLE_DEVICES=1 nohup python -u train_classifier_with_augmented_images.py --dataset animals --episodes_number 10 --selected_classes 5  --classification_total_epoch 50  --experiment_title AugmentedClassifier_animals_augmented512  --image_width 128  --image_height 128 --image_channel 3  > 22.log 2>&1 & 



CUDA_VISIBLE_DEVICES=0 python train_classifier_with_augmented_images.py --dataset flowers --episodes_number 10 --selected_classes 5  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedClassifier_flowers_augmented512  --image_width 128  --image_height 128 --image_channel 3
#############################
sequence running For animals
#############################




