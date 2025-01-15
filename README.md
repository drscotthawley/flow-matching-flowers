# flow-matching-flowers
An exercise in flow-matching modeling of the Oxford Flowers dataset

This is mostly code adapted by @drscotthawley from [Tadao Yamaoka's CIFAR10 Example](https://tadaoyamaoka.hatenablog.com/entry/2024/10/09/232749). As of now, this repo is purely an academic self-education exercise.

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

*..
*..
* https://github.com/YangLing0818/consistency_flow_matching

...and I may execute and/or run those as needed. Yamaoka's was the first code I found online so I went with that.  
(I do have my own code, but I was having problems so I started looking for other implementations!). 


# Sample Results

See WandB pages for different resolutions: 

* 32x32 (easy): https://wandb.ai/drscotthawley/TadaoY-flowers-32x32?nw=nwuserdrscotthawley
* 64x64 (also pretty easy): https://wandb.ai/drscotthawley/TadaoY-flowers-64x64?nw=nwuserdrscotthawley
* 128x128 (??? hard): https://wandb.ai/drscotthawley/TadaoY-flowers-128x128?nw=nwuserdrscotthawley

Re. scaling up, working in some kind of VAE space like the [flowers-vqgan](https://github.com/drscotthawley/vqgan-shh) I made last fall could help with resolution, but... actually I was having trouble doing that with another dataset (128x128x3 images compressed to 16x16x4 RVQ), so... this is why I'm back to the flowers dataset.)
