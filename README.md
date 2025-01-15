# flow-matching-flowers
An exercise in flow-matching modeling of the Oxford Flowers dataset

This is mostly code adapted by @drscotthawley from [Tadao Yamaoka's CIFAR10 Example](https://tadaoyamaoka.hatenablog.com/entry/2024/10/09/232749). As of now, this repo is purely an academic self-education exercise.

This repo was created to share progress and solicit feedback, advice, comments, etc. for improving such a code for examples beyond 2D dots and tiny images, e.g. I'm having trouble with 128x128 images. Please use the [Discussions](https://github.com/drscotthawley/flow-matching-flowers/discussions) feature to do so, and thank you! 



# Installation 
```bash
git clone https://github.com/drscotthawley/flow-matching-flowers.git
cd flow-matching-flowers
pip install -r requirements.txt
# optional: pip install -e . 
```

# Execution
```
python3 ./train_flowers.py
```

# Warning
This is still  a bit janky (AF) so the code is the documentation.  I mainly made this repo to facilitate soliciting HELP from others! ;-) 


# Related Work/Repos

There are other repo's on GitHub on this topic, e.g. 

* https://github.com/851695e35/Flow-Matching-Implementation

* https://github.com/facebookresearch/flow_matching

* https://github.com/YangLing0818/consistency_flow_matching

...and I may execute and/or run those as needed. Yamaoka's was the first code I found online so I went with that.  
(I do have my own code, but I was having problems so I started looking for other implementations!). 


# Sample Results

See WandB pages for different resolutions: 

* 32x32 (easy): https://wandb.ai/drscotthawley/TadaoY-flowers-32x32?nw=nwuserdrscotthawley
* 64x64 (also pretty easy): https://wandb.ai/drscotthawley/TadaoY-flowers-64x64?nw=nwuserdrscotthawley
* 128x128 (??? hard: converges super-slowly, numerical instabilities, pictures look bad): https://wandb.ai/drscotthawley/TadaoY-flowers-128x128?nw=nwuserdrscotthawley
  * Adding weight decay slows loss but doesn't prevent instabilities.
  * EMA weights may help. TODO. 
  * Combining K&Q embeddings as per Stable Diffusion 3 could help with instability. TODO. 


# Scaling Beyond 64x64?

Re. scaling up, alternate ideas are possible:

1. working in some kind of compressed (VQ-)VAE space like the [flowers-vqgan](https://github.com/drscotthawley/vqgan-shh) I made last fall could help with, but... actually I was having trouble doing that with another dataset (128x128x3 images compressed to 16x16x4 RVQ), so... this is why I'm back to trying a more standard dataset (the flowers), and with no compression. (Trying the VQ thing is next on my to-do list though.)
2. Doing a multiscale or progressive-multiscale approach ala Google's ImageGen, using the endpoint of a lower-resolution flow, upscaling it, and training a larger model to predict the difference.  Relatedly, Stable Diffusion 3's base model was 128x128 upon which they trained super-resolution models.  Maybe the key for 128x128 is just to let it train for months? 

# Improving the "rate of training"?

The graph of Loss function vs steps for learning the velocity field tends to flatten out over time, but still the resulting output images look "bad". Presumably this is because tiny errors in the velocity field can integrate to large errors in the generated output, i.e. being off the target data manifold. Even when using fancy integration schemes and many steps. 

There is typically no "reconstruction loss" with flow-matching (or diffusion) models. Trying to include one would mean storing gradients throughout the integration process, which seems infeasible.

How else can we improve the rate at which the model learns to produce good outputs? I've tried two methods, neither of which has helped much. 

1. Trying different interpolation schemes (e.g. "cosine" instead of the standard linear interoplation for flow-matching) can produce "straigher" trajectories but I **haven't observed it making much difference in the outputs.**  (And re. straightening: ReFlow's not appropriate yet because so far our trained endpoints look bad.)

2. Trying to pair the source & target points "intelligently"* instead of random pairing can result in straigter trajectories by eliminating many "incorrect" trajectories, however in my experience these methods, while resulting in lower loss values, don't actually "make much difference", i.e don't improve the integrated output distributions *or even the rate of convergence of independent metrics* as the model trains. 

​    * e.g. via sorting (in 1D) or partitioning space using  indices of vector quantization (in high D). One could imagine other partitioning schemes such as k-means, Locality Sensitive Hashing, Barnes-Hut, etc..  but... so far it hasn't made a difference?
