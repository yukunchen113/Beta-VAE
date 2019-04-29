from run_model import main
import general_utils as gu
import os
import numpy as np
import general_constants as gc
#train with different KLD modifiers
#regular VAE
#main(is_train=True, save_folder="VAE", kld_function=lambda **kwargs: 1)
#beta = 10
#main(is_train=True, save_folder="10-Beta-VAE", kld_function=lambda **kwargs: 10)
#beta = 100
#main(is_train=True, save_folder="100-Beta-VAE", kld_function=lambda **kwargs: 100)
#beta = 1000
#main(is_train=True, save_folder="1000-Beta-VAE", kld_function=lambda **kwargs: 1000)
#beta = 10000
#main(is_train=True, save_folder="10000-Beta-VAE", kld_function=lambda **kwargs: 10000)
#KLD annealing to 10 with 40000 steps
main(is_train=True, save_folder="KLD_Anneal_40000_steps-10-Beta-VAE", kld_function=lambda step,**kwargs: min(10, step/4000), validation_off=True, max_steps = 50000)
#KLD decreasing to 0 from 100 in 40000 steps
#main(is_train=True, save_folder="KLD_Decrease_40000_steps-100to1-Beta-VAE", kld_function=lambda step,**kwargs: max(1, 100-step/400), validation_off=True, max_steps = 50000)
#KLD decreasing to 0 from 10000 in 100000 steps
#main(is_train=True, save_folder="KLD_Decrease_100000_steps-10000to1-Beta-VAE", kld_function=lambda step,**kwargs: max(1, 10000-step/10), validation_off=True, max_steps = 125000)

#run the models
dataset, get_group = gu.get_celeba_data(gc.datapath)#, preprocess_fn = lambda x: np.array(Image.fromarray(x).resize((x.shape[0], *imreshape_size, x.shape[-1]))))
test_images, test_labels = get_group(group_num=5, random_selection=True, remove_past=True)  # get images, and remove from get_group iterator
#test_images
#test_latents
#modelsavedir
logdir = "predictions"
modelsavedir = [os.path.join(logdir, i, "model_save_point") for i in os.listdir(logdir)]

print(modelsavedir)
main(is_train=False, test_images=test_images, modelsavedir=modelsavedir, save_image_results=None, return_images=True)

#center = np.asarray([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
#test_latents = np.concatenate((center, np.eye(32)),axis=0)
#main(
#	is_train=False, 
#	test_latents=test_latents, 
#	modelsavedir="predictions/test_file/model_save_point/", save_image_results=None)
