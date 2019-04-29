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
#main(is_train=True, save_folder="KLD_Anneal_40000_steps-10-Beta-VAE", kld_function=lambda step,**kwargs: min(10, step/4000), validation_off=True, max_steps = 50000)
#KLD decreasing to 0 from 100 in 40000 steps
#main(is_train=True, save_folder="KLD_Decrease_40000_steps-100to1-Beta-VAE", kld_function=lambda step,**kwargs: max(1, 100-step/400), validation_off=True, max_steps = 50000)
#KLD decreasing to 0 from 10000 in 100000 steps
#main(is_train=True, save_folder="KLD_Decrease_100000_steps-10000to1-Beta-VAE", kld_function=lambda step,**kwargs: max(1, 10000-step/10), validation_off=True, max_steps = 125000)


#"""
# traversing the latent space:
def create_interpolations(latent_space_anchor, latent_direction_1, latent_size, grid_size = 8):
	#create grid of interpolations between faces:
	v1 = (latent_direction_1 - latent_space_anchor)/(grid_size-1)
	axis1 = np.arange(grid_size).reshape(-1,1,1)*v1
	generation_latent_space = axis1+latent_space_anchor
	return generation_latent_space.reshape(-1,latent_size)

#run the models
dataset, get_group = gu.get_celeba_data(gc.datapath)#, preprocess_fn = lambda x: np.array(Image.fromarray(x).resize((x.shape[0], *imreshape_size, x.shape[-1]))))
test_images, test_labels = get_group(group_num=1, random_selection=True, remove_past=True)  # get images, and remove from get_group iterator
#test_images
#test_latents
#modelsavedir
logdir = "predictions"
modelsavedir = [os.path.join(logdir, i, "model_save_point") for i in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, i))]

return_dict = main(is_train=False, test_images=test_images, modelsavedir=modelsavedir, save_image_results=None, return_images=True)
images_path = []
for model, latent_data in return_dict.items():
	mean, std, kld = latent_data
	print()
	print(model)
	mean_sorted = np.argsort(np.abs(mean), axis=0).tolist()
	#kld_sorted = np.argsort(kld, axis=0).tolist()
	latent_space_anchor = np.zeros((32))
	latent_direction_1 = np.zeros((32))
	latent_space_anchor[mean_sorted[-1]] = -3
	latent_direction_1[mean_sorted[-1]] = 3
	num_images = 10
	latents = create_interpolations(latent_space_anchor, latent_direction_1, 32, num_images)
	latents = np.stack((latents, latents+std, latents+std*2))
	print(latents.shape)

	latents = latents.transpose(1,0,2).reshape(-1, 32)
	return_dict = main(is_train=False, test_latents=latents, modelsavedir=["/".join(model.split("/")[:-1])], 
		save_image_results=os.path.join("images", "%s.jpg"%model.split("/")[1]), return_images=False, save_images_aspect_ratio=[3,num_images])
	images_path.append(os.path.join("images", "%s.jpg"%model.split("/")[1]))
print(images_path)
#"""