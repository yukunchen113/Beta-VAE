import tensorflow as tf
import numpy as np
import general_constants as gc
import general_utils as gu
import os
import time
import matplotlib.pyplot as plt
from functools import reduce
import scipy.ndimage as ndi
import shutil
import params as pm
from PIL import Image
from PIL import ImageDraw
import model as md
from PIL import ImageFont
np.set_printoptions(suppress=True)

def main(is_train=False, **kwargs):
	tf.reset_default_graph()
	params = pm.general_params
	batch_size = params["batch_size"]
	log_step = params["log_step"]
	num_steps = params["num_steps"]
	learning_rate = params["learning_rate"]
	plot_step = params["plot_step"]
	validation_step = params["validation_step"]
	latent_size = params["latent_size"]
	initializer_step = params["initializer_step"]
	imreshape_size = params["imreshape_size"]
	validation_tolerence = params["validation_tolerence"]
	data_shape = params["data_shape"]
	if not is_train:
		#when testing, the following keyword arguments must be specified.
		assert "test_images" in kwargs or "test_latents" in kwargs, "must specify a latent space or test images for reconstruction"
		assert "modelsavedir" in kwargs, "must specify path for saved models"
		if "test_images" in kwargs:
			test_images = kwargs["test_images"]
			if test_images.ndim < 4:
				test_images = np.expand_dims(test_images, 0)
		else:
			test_images = None

		if "test_latents" in kwargs:
			test_latents = np.asarray(kwargs["test_latents"])
			if test_latents.ndim < 2:
				test_latents = np.expand_dims(test_latents, 0)
		else:
			test_latents = None
		for i in kwargs["modelsavedir"]:
			assert "checkpoint" in os.listdir(i), "no saved model in one of the specified directories:%s"%i
		modelsavedir = [os.path.join(i, "model.ckpt") for i in kwargs["modelsavedir"]]

	if "max_steps" in kwargs:
		num_steps = kwargs["max_steps"]

	if is_train:
		# if specified, create general path (if doesn't exist) 
		# Then, define model save path
		if "save_folder" in kwargs:
			pm.create_new_path(pm.logdir, False)
			logdir = os.path.join(pm.logdir, kwargs["save_folder"])
		else:
			logdir = pm.logdir

		# delete previous model path if training.
		pm.create_new_path(logdir, is_train)

		imgdir = os.path.join(logdir, "images")

		# delete previous image path if training.
		pm.create_new_path(imgdir, is_train)

		modelsavedir = os.path.join(logdir, "model_save_point")
		pm.create_new_path(modelsavedir, False)

		log_file = os.path.join(logdir, "log.txt")
		if os.path.exists(log_file) and is_train:
			os.remove(log_file)

		# copy parameters to predictions.
		for file_to_be_copied in ["params.py", "model.py", "main.py"]:
			shutil.copyfile(file_to_be_copied, os.path.join(logdir, file_to_be_copied))

	# get data
	if is_train:
		dataset, get_group = gu.get_celeba_data(gc.datapath)#, preprocess_fn = lambda x: np.array(Image.fromarray(x).resize((x.shape[0], *imreshape_size, x.shape[-1]))))
		test_images, test_labels = get_group(group_num=initializer_step//1000, random_selection=False, remove_past=True)  # get images, and remove from get_group iterator
		validation_images, validation_labels = get_group(group_num=1, random_selection=False, remove_past=True)  # get images, and remove from get_group iterator

	data_shape = data_shape if test_images is None else list(test_images.shape)
	data_shape[0] = batch_size

	# create placeholders
	inputs_ph = tf.placeholder(tf.float32, shape=(None, *data_shape[1:]), name="inputs_ph")  # these are the ones fed into the network, after been batched.
	outputs_ph = tf.placeholder(tf.float32, name="outputs_ph")  # these are the ones fed into the network, after been batched.
	#inputs_set_ph = tf.placeholder(tf.float32, name="inputs_set_ph")  # these are the ones fed into the iterator, to be batched.
	#outputs_set_ph = tf.placeholder(tf.float32, name="outputs_set_ph")  # these are the ones fed into the iterator, to be batched.
	#iterator, next_element = gu.get_iterator(batch_size, inputs=inputs_set_ph, labels=outputs_set_ph)  # this is the iterator.

	# preprocessing
	inputs = inputs_ph
	# crop to 128x128 (centered), this number was experimentally found
	image_crop_size = [128,128]
	inputs=tf.image.crop_to_bounding_box(inputs, 
		(inputs.shape[-3]-image_crop_size[0])//2,
		(inputs.shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	inputs = (tf.image.resize_images(inputs, imreshape_size, True))
	inputs = params["preprocess_inputs"](tf.clip_by_value(inputs, 0, 255))

	# make model
	vae = md.VariationalAutoEncoder(inputs, pm.model_params)
	dist_params = vae.get_latent_distribution()  # get information about the latent distribution.
	reconstruction_prediction = vae.get_reconstruction()
	latents_ph, generation_prediction = vae.get_generation()

	# get loss
	reconstruction_loss = vae.reconstruction_loss(inputs)
	reconstruction_loss = tf.reduce_mean(tf.abs(reconstruction_loss))
	regularization_loss = tf.reduce_mean(vae.kl_isonormal_loss())
	kl_multiplier = tf.placeholder(tf.float32, name="kl_multiplier")
	loss = reconstruction_loss+kl_multiplier*regularization_loss

	# training:
	train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)

	#save model:
	saver = tf.train.Saver()

	# latent space analysis:
	
	# min, max means and standard deviations across the 
	std_analysis = [tf.reduce_mean(tf.exp(0.5*dist_params[1])), tf.reduce_min(tf.exp(0.5*dist_params[1])), tf.reduce_max(tf.exp(0.5*dist_params[1]))]
	mean_analysis = [tf.reduce_mean(dist_params[0]), tf.reduce_min(dist_params[0]), tf.reduce_max(dist_params[0])]

	latent_element_std_average = tf.reduce_mean(tf.exp(0.5*dist_params[1]), axis=0)
	latent_element_mean_average = tf.reduce_mean(dist_params[0], axis=0)
	latent_element_kld_average = tf.reduce_mean(vae.kl_isonormal_loss(False), axis=0)
	latent_element_max = tf.reduce_max(vae.latent_output, axis=0)
	latent_element_min = tf.reduce_min(vae.latent_output, axis=0)

	# run model
	with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
		# print(training_data["data"].shape)
		sess.run(tf.global_variables_initializer())
		if is_train:
			test_images = test_images[:batch_size]

			validation_feed_dict = {  # keep all these images to test constant.
				inputs_ph: validation_images,
				outputs_ph: validation_labels,
			}
			prev_validation_loss = None
			current_validation_count = 0
		else:
			num_steps = len(modelsavedir)
			return_dict = {}
		first_step = True
		if is_train:
			log_file = open(log_file, "a")
		for step in range(num_steps):
			if not is_train:
				saver.restore(sess, modelsavedir[step])
			if is_train:
				if not step%(initializer_step//batch_size):
					#get images and labels
					images, labels = get_group(group_num=initializer_step//1000, random_selection=True)
					images = images[:images.shape[0]//batch_size*batch_size] #cut to batch_size
					labels = labels[:labels.shape[0]//batch_size*batch_size]
					images, labels = gu.shuffle_arrays(images, labels)


				feed_dict = {
					inputs_ph:images[batch_size*(step%(initializer_step//batch_size)):batch_size*(
							step%(initializer_step//batch_size)+1)],
					outputs_ph:labels[batch_size * (step % (initializer_step // batch_size)):batch_size * (
								step % (initializer_step // batch_size) + 1)]
				}


				regularization_loss_val, reconstruction_loss_val = sess.run([regularization_loss, reconstruction_loss],
																			feed_dict = feed_dict)
				train_feed = feed_dict.copy()


				kl_mul_val = 1 if not "kld_function" in kwargs else kwargs["kld_function"](step=step)
				train_feed[kl_multiplier] = kl_mul_val

				loss_val,_ = sess.run([loss, train_op], feed_dict=train_feed)

				print_out = "step: %d, \ttotal loss: %.3f, \tRegularization loss: %.3f, \tReconstruction loss: %.3f, kl weight: %f,\
				\n Latent Space Analysis: \naverage stddev %s,\t stddev range [%s,	 \t%s], \
				\naverage mean %s,   \tmean range [%s,	\t%s]"%(step, loss_val, regularization_loss_val, reconstruction_loss_val, train_feed[kl_multiplier], *sess.run([*std_analysis, *mean_analysis], feed_dict=feed_dict))
				print(print_out)
				print(""%sess.run([], feed_dict=feed_dict))
				log_file.write("%s\n"%print_out)
				if np.isnan(loss_val):
					break

				if not step%validation_step:
					validation_feed_dict[kl_multiplier] = kl_mul_val
					validation_loss_val = sess.run(loss, feed_dict=validation_feed_dict)
					if prev_validation_loss is None or validation_loss_val < prev_validation_loss or ("validation_off" in kwargs and kwargs["validation_off"]):
						current_validation_count = 0
						#save model
						saver.save(sess, os.path.join(modelsavedir, "model.ckpt"))

						prev_validation_loss = validation_loss_val
					else:
						current_validation_count+=1
						if current_validation_count > validation_tolerence:
							break


			if is_train and (not step%plot_step or step in log_step):
				if first_step:
					test_feed_dict = {  # keep all these images to test constant.
						inputs_ph: test_images,
						outputs_ph: test_labels[:batch_size],
					}
					first_step = False

				#create grid of interpolations between faces:
				grid_size = [6,8]
				latent_space_generation = sess.run(vae.latent_output, feed_dict=test_feed_dict)[:3]
				v1 = (latent_space_generation[1] - latent_space_generation[0])/(grid_size[0]-1)
				v2 = (latent_space_generation[2] - latent_space_generation[0])/(grid_size[1]-1)
				axis1 = np.arange(grid_size[0]).reshape(-1,1,1)*v1
				axis2 = np.arange(grid_size[1]).reshape(1,-1,1)*v2
				generation_latent_space = axis1+axis2+latent_space_generation[0]
				generation_latent_space = generation_latent_space.reshape(-1,latent_size)
				test_feed_dict[latents_ph] = generation_latent_space

				
				#save image of data:
				#create reconstruction
				orig_images_val, recon_val, gener_val = sess.run([inputs, reconstruction_prediction, generation_prediction], feed_dict=test_feed_dict)
				
				#print("MIN, MAX", np.amin(recon_val), np.amax(recon_val))
				save_image_results = True
				if "save_image_results" in kwargs:
					save_image_results = kwargs["save_image_results"]
				create_images(step, imgdir, params["postprocess_outputs"], orig_images_val[:48], recon_val[:48], gener_val[:48])
			
			if not is_train:
				test_feed_dict = {}
				create_images_kwargs = {}
				if not test_images is None:
					test_feed_dict[inputs_ph] = test_images
					orig_images_val, recon_val = sess.run([inputs, reconstruction_prediction], feed_dict=test_feed_dict)
					create_images_kwargs["original_images"] = orig_images_val[:48]
					create_images_kwargs["reconstruction"] = recon_val[:48]
					latent_analysis = sess.run([
						latent_element_mean_average, 
						latent_element_std_average, 
						latent_element_kld_average,
						latent_element_max,
						latent_element_min,], feed_dict=test_feed_dict)
					#print()
					#print(modelsavedir[step].split("/")[1])
					return_dict[modelsavedir[step]] = latent_analysis
					#latent_analysis = np.transpose(latent_analysis)
					#for i in range(len(latent_analysis)):
					#	print("Latent element %d: mean, std, kld\t"%i, latent_analysis[i])

				if not test_latents is None:
					test_feed_dict[latents_ph] = test_latents
					gener_val = sess.run(generation_prediction, feed_dict=test_feed_dict)
					create_images_kwargs["generation"] = gener_val[:48]
				save_image_results = True
				if "save_image_results" in kwargs:
					save_image_results = kwargs["save_image_results"]
				save_images_aspect_ratio = None
				if "save_images_aspect_ratio" in kwargs:
					save_images_aspect_ratio = kwargs["save_images_aspect_ratio"]

				if not ("return_images" in kwargs and kwargs["return_images"]):
					create_images(step, imgdir=save_image_results, postprocess_outputs=params["postprocess_outputs"], save_images_aspect_ratio=save_images_aspect_ratio, **create_images_kwargs)
				else:
					pass


		if is_train:
			log_file.close()
		else:
			return return_dict

def create_images(step, imgdir, postprocess_outputs, save_images_aspect_ratio=None, original_images=None, reconstruction=None, generation=None):
	"""
	Creates 3 sets of images for VAE, reconstruction, generation, and original
	:param step: This is the current step of training. This will also be part of the name for the file.
	:param original_images: This is an array of original images, the ground truth
	:param reconstruction: This is the reconstructed images, from original_images.
	:param generation: This is the generated images,
	:return: None
	"""
	images_type = []
	possible_captions = ["reconstruction original images", "image reconstruction", "image generation"]
	captions = []
	j = 0
	for i in [original_images, reconstruction, generation]:
		if not i is None:
			height, width = gu.find_largest_factors(len(i))
			aspect_ratio = save_images_aspect_ratio if not save_images_aspect_ratio is None else [height, width]
			images_type.append(gu.create_image_grid(i, aspect_ratio))
			captions.append(possible_captions[j])
		j+=1
	# create images
	im = []
	header_size = 30  # amount of space for caption
	for i in range(len(images_type)):
		image_type = images_type[i]
		#image_type = np.log(5*(image_type+1))
		#image_type = image_type-np.amin(image_type)
		#image_type = image_type/np.amax(image_type)
		caption = captions[i]
		container = np.squeeze(np.ones((image_type.shape[0]+header_size, *image_type.shape[1:])))
		container[:-header_size] = image_type
		#print(np.uint8(postprocess_outputs(container)).dtype)
		im.append(Image.fromarray(np.uint8(postprocess_outputs(container))))
		#print("MIN, MAX", np.amin(container), np.amax(container))
		ImageDraw.Draw(im[i]).text((5,image_type.shape[0]+2), caption)

	width = max([i.size[0] for i in im])
	height = sum([i.size[1] for i in im])
	header = 40
	margin = 20

	if len(images_type[0].shape) == 3:
		n_channels = images_type[0].shape[2]
		total_image = Image.fromarray(np.ones((height+header, width+margin, n_channels), dtype=np.uint8)*255)
	else:
		total_image = Image.fromarray(np.ones((height+header, width+margin), dtype=np.uint8)*255)

	for i in range(len(im)):
		image = im[i]
		total_image.paste(image, (margin//2,header+i*image.size[1]))

	ImageDraw.Draw(total_image).text((margin//2+10,5), "Step: %d"%step)
	total_image.convert('RGB')
	if not imgdir is None:
		if not ".jpg" in imgdir:
			imgdir = os.path.join(imgdir, "image_%s.jpg"%step)
		total_image.save(imgdir)
	else:
		total_image.show()
		input()
if __name__ == "__main__":
	main()