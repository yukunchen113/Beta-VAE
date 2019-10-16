# β-VAE

## About this Project
This project implements the β-VAE. β is a term which controls disentanglement within latent representations.

Please see my results on [my article on β-VAEs](https://yukunchen.me/project/2019/06/09/Beta-Variational-Autoencoder.html)

Please see my project on implementing a [VAE](https://github.com/yukunchen113/VariationalAutoEncoder) for more on VAE latent space analysis.

Model checkpoints, parameters files and run files are saved in the predictions folder. The model files are called model.ckpt



## Requirements
Please pull my utils_tf1 repo, and add it to your python path. The functions there are used.

The model training is contained in main.py, the parameters are in params.py and the predictions folder contains saved images during training. The models.py file contains the model for the VAE
