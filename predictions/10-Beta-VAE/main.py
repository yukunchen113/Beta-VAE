from run_model import main
import os

#run different KLD modifiers

#regular VAE
main(save_folder="VAE", kld_function=lambda **kwargs: 1)

#beta = 10
main(save_folder="10-Beta-VAE", kld_function=lambda **kwargs: 10)

#beta = 100
main(save_folder="100-Beta-VAE", kld_function=lambda **kwargs: 100)

#beta = 1000
main(save_folder="100-Beta-VAE", kld_function=lambda **kwargs: 1000)

#beta = 10000
main(save_folder="100-Beta-VAE", kld_function=lambda **kwargs: 10000)

#KLD annealing to 10 with 40000 steps
main(save_folder="KLD_Anneal_40000_steps-10-Beta-VAE", kld_function=lambda step,**kwargs: min(10, step/40000), validation_off=True, max_steps = 50000)

#KLD decreasing to 0 from 100 in 40000 steps
main(save_folder="KLD_Decrease_40000_steps-100to0-Beta-VAE", kld_function=lambda step,**kwargs: min(0, 100-step/400), validation_off=True, max_steps = 50000)

#KLD decreasing to 0 from 10000 in 100000 steps
main(save_folder="KLD_Decrease_100000_steps-10000to0-Beta-VAE", kld_function=lambda step,**kwargs: min(0, 10000-step/10), validation_off=True, max_steps = 125000)
