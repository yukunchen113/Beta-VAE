import os
import general_utils as gu
import tensorflow as tf
import shutil
# set paths
logdir = "predictions"
tblogdir = "tensorboard_logdir"


def create_new_path(path, del_prev=True):
	if not os.path.exists(path):
		os.mkdir(path)
	else:
		if del_prev:
			shutil.rmtree(path)
			os.mkdir(path)


training_params = {
	# model data params
	"batch_size": 64,  # the size for batch training

	# for plotting/logging
	"plot_step": 1000,  # will save the plot every this many steps.
	"log_step": [0, 1, 3, 7, 10, 16, 30, 100, 300, 500, 700, 1500, 
		2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500],  # additional plot saves on these steps

	# for training
	"learning_rate": 0.0005,  # learning rate
	"num_steps": 100000,  # the total number of steps to take when training
	"initializer_step": 5000,  # the number of steps to take before reinitialization
	"validation_step":500,#apply validation step every this many steps.
	"validation_tolerence":20 # wait this long for validation
}

general_params = {
	#for image preprocessing
	"imreshape_size": [64, 64],  # this is the shape of the image to be resized to
	"data_shape": [None, 218, 178, 3], #this is the shape of the images
}

model_params = {
	"encoder_layers":[[32,4,2], [32,4,2],[32,4,2],[32,4,2],[256],[256]],

	"latent_size":32,

	"decoder_layers":None,

	"decoder_activation":tf.nn.sigmoid,

	# for the loss
	"loss_type": [
		gu.cross_entropy,
		lambda inputs, prediction: tf.reduce_sum(tf.losses.mean_squared_error(inputs, prediction,reduction=tf.losses.Reduction.NONE), axis=list(range(1,len(inputs.shape)))),
		lambda inputs, prediction: tf.reduce_sum(tf.losses.absolute_difference(inputs, prediction,reduction=tf.losses.Reduction.NONE), axis=list(range(1,len(inputs.shape)))),
		][1],  # choose one of these losses.
}

# apply preprocessing on model activations:
if model_params["decoder_activation"] == tf.sigmoid:
	model_params["preprocess_inputs"] = lambda x: x/255
	model_params["postprocess_outputs"] = lambda x:x*255
elif model_params["decoder_activation"] == tf.tanh:
	model_params["preprocess_inputs"] = lambda x: x/127.5-1
	model_params["postprocess_outputs"] = lambda x:(x+1)*127.5
else:
	model_params["preprocess_inputs"] = lambda x: x
	model_params["postprocess_outputs"] = lambda x: x

general_params.update(training_params)
general_params.update(model_params)