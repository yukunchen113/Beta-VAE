# β-VAE

## About this Project
This project implements the β-VAE. β is a term which controls disentanglement within latent representations.

Please see my project on implementing a [VAE](https://github.com/yukunchen113/VariationalAutoEncoder) for more on VAE latent space analysis.

Model checkpoints, parameters files and run files are saved in the predictions folder. See _Analysis_ section below for more on each run. The model files are called model.ckpt

## Requirements
Please pull my utils repo, and add it to your python path. The functions there are used.

The model training is contained in main.py, the parameters are in params.py and the predictions folder contains saved images during training. The models.py file contains the model for the VAE

## β-VAE background
### Resources
These are resources for VAEs and β-VAE:
- VAE Original Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- VAE Explanation: [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- β-VAE Original Paper: [β-VAE](https://openreview.net/pdf?id=Sy2fzU9gl)
- Understanding disentangling in β-VAE: [Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf)

### From VAE to β-VAE
In terms of architecture, β-VAE is the same as the VAE. The change is in the loss function. the β term is a multiplier on the KLD term. 

Recall how each latent representation given a training sample is a probability distribution. An increase in the β term will cause the probabilites to become more like a zero-centered isotropic normal. The standard deviations will become closer to 1, and the means will be closer to 0. The natural form that the loss without the KLD term will want to take will be, dirac deltas that are spread out (have a large range of means and close to 0 standard deviation). This is to prevent the learned representation of each image from affecting each other as much as possible. Increasing the β term will squish the distributions, and will widen each distribution. Forcing there to be significant representations inbetween points. This is the VAE objective.

### Theory

#### How disentanglement happens in β-VAE and how it relates to the VAE objective
By Squishing all the distributions of the latent representation together, the representations are forced to share common features to maintain accuracy. The representations will start to represent the most promenent features (according to the loss), maximizing the use of the limited capacity due to the constraints of the KLD term.

The limited capacity forces disentanglement, where the parts that are common across models. Through the architecture, the covariance matrix of the latent representation is also constrained to be a diagonal matrix, forcing basis of the latent representations to align with each element in the latent representation. This causes the parts of variance to be aligned with the each element in the latent representation.

For more on this theory, please see [Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf).

#### How does disentanglement correspond with the features we recognize?

But how does the specific disentangled features correspond to the features that we recognize? My thoughts are as follows. First off, higher sources of variance will require information to be passed through. [Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf) explains that changes in features that inquire greater amounts of loss will be more heavily valued/biased towards. This means that when using MSE as a loss, position of the image is most likely to be learned first. This is due to the fact that a slightly shifted reconstruction image would have a high loss, even if it is perfect. High variance in one direction (high variance of one feature) would allow the representation of that feature to be more disentangled. This is to maximize the amount of information that is transfered about that feature for accurate reconstructions, to decrease the loss given the wide range of different faces. High variance which have little effect on the loss might be ignored by the model to prioritize other features. Low variance and high loss might be represented depending on how much variance there is in the feature and how much loss is effected. Very low amounts of variance would cause the model to just memorize the mean across the feature, given that this will minimize the loss to a satisfactory result. The variance mentioned here is the diversity of different examples given a certain feature, which I will call, the variance of a feature. The variance of a feature and the amount of loss a small change in that feature causes can be summarized by the amount of variance of the feature from the point of view of the loss (I will call this _loss feature variance_), which is a direct way of calculating the probability that a feature will be disentangled. High amounts of loss feature variance will cause high amounts of loss if this feature is not learned properly.

The reason why disentanglement 'happens' to correspond to the features that we recognize is due to the fact that there is structure within the data, which can be found by both humans and AI. I mentioned above that a function of the variance in the features, given the bias in the loss is what causes disentanglement in the model. I believe that this _variance in the features_ is what people use to classify different features. If there is an variance between similar details (these details are mutually exclusive with each other), then this will be called a feature. differnt types of features are separated by further differences, and are independent/have cases which they are independent of each other. For example, if eye color is a feature, it can only have one dominant colour (from a high level view of the face, for simple models). Whereas nose shape is independent of eye color will be a separate feature. (of course, we can get more specific with multieye colour, but even then, each colour would then be a feature, still maintaining independence). The model learns these facts though the loss as defined above.

Something interesting about disentanglement is it's relationship with domain randomization. Domain randomization seems to increase the amount of data used for training. It seems to directly try to tackle the the problem above, learning variance between the features (which covary, which ones are independent and which ones are mutually exclusive) by directly increasing the amount of variance per feature, which will further distinguish each feature. Domain randomization aims to directly increase the amount of variance between desirable features in the data, letting the model to find the structure within the relationships of the data, while causing a natural regularizing effect with noise withing the original data generative factors. Disentanglement using VAEs lower the information capacity of what is being passed through as a regularizer, which will cause the model to be more sensitive to these high amounts of variance thus causing a similar effect to domain randomization. Both these regularizing effects aim for stability to different features/examples and learning of the data generative factors. Though the difference between them seems to be how they treat the nuance factors. Domain randomization aims to ignore them, and disentanglement aims to still take them into account. (invariance vs equivariance)

### Disentanglement Metric
TBD

## Analysis:

See the images/Training Results folder for various results ran by different training methods. These results are in the same format as the results from my VAE project. For this project, I aim to explore the new things that disentanglement brings to the table.

Here are the tests that were run:
- regular VAE: VAE
- β set to 10: 10-Beta-VAE
- β set to 100: 100-Beta-VAE
- β set to 1000: 1000-Beta-VAE
- β set to 10000: 10000-Beta-VAE
- KLD β annealing to 10 with 40000 steps: KLD_Anneal_40000_steps-10-Beta-VAE
- KLD β decreasing to 0 from 100 in 40000 steps: KLD_Decrease_40000_steps-100to1-Beta-VAE
- KLD decreasing to 0 from 10000 in 100000 steps: KLD_Decrease_100000_steps-10000to1-Beta-VAE

To do analysis of disentanglement, we should decide which element in the latent representation to vary. Lets try the most distinct feature first. Under heavy regularizations, the model will try to optimize for the feature that affects the loss the most. We can try to get the average distribution per element that is the furthest from a zero-centered isotropic normal distribution, specifically, the mean should be the farthest from zero. Lets try to see what happens when varying this element.

Here I used a zero vector with the ith element as a value from range -3 to 3, where i is the element with the highest absolute mean. Keep in mind that this might be a source of error, as the space of the training latents is shrinked as the β term is increased. This means that the range is not relative to the range of training samples.

![β = 10000, KLD is too large, causing posterior collapse](images/latent_traversal/10000-Beta-VAE.jpg)
![β = 1000, KLD is too large, causing posterior collapse](images/latent_traversal/1000-Beta-VAE.jpg)
![β = 100, seems to be traversing background colour and shading, it counts hair colour as part of background](images/latent_traversal/100-Beta-VAE.jpg)
![β = 10, seems to be traversing background colour and gender](images/latent_traversal/10-Beta-VAE.jpg)
![KLD β decreasing to 0 from 100 in 40000 steps, here the face structure is changing, specifically the jaw and forehead](images/latent_traversal/KLD_Decrease_40000_steps-100to1-Beta-VAE.jpg)
![KLD β annealing to 10 with 40000 steps, The background colour and shading is changing](images/latent_traversal/KLD_Anneal_40000_steps-10-Beta-VAE.jpg)
![KLD decreasing to 0 from 10000 in 100000 steps, The hair colour and azimuth is changing as well.](images/latent_traversal/KLD_Decrease_100000_steps-10000to1-Beta-VAE.jpg)
![regular VAE, the gender, smile, and azimuth is changing here.](images/latent_traversal/VAE.jpg)

