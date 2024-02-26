# Neural Microfacet Fields for Inverse Rendering
More details can be found at the project page [here](https://half-potato.gitlab.io/posts/nmf/).
# Installation
A conda virtual environment is recommended.
```
pip install -r requirements.txt
```
Dataset dir should contain a folder named `nerf_synthetic` with various datasets in the `blender` configuration.
```
python train.py -m expname=v38_noupsample model=microfacet_tensorf2 dataset=ficus,drums,ship,teapot vis_every=5000 datadir={dataset dir}
```
Experiment configurations are done using hydra, which controls the initialization parameters for all of the modules. Look in `configs/model` to 
see what options are available. Setting the BRDF activation would look like adding this:
```
model.arch.model.brdf.activation="sigmoid"
```
to the command line argument.

To relight a dataset, you need to first convert the environment map .exr file to a pytorch checkpoint `{envmap}.th` like this:
```
python -m scripts.pano2cube backgrounds/christmas_photo_studio_04_4k.exr --output backgrounds/christmas.th
```
Then, after training some model and obtaining a checkpoint `{ckpt}.th`, you can run
```
python train.py -m expname=v38_noupsample model=microfacet_tensorf2 dataset=ficus vis_every=5000 datadir={dataset dir} ckpt={ckpt}.th render_only=True fixed_bg={envmap}.th
```

# Recreating Experiments
Note that something is currently wrong with computation of metrics in the current code and the scripts `reval_lpips.ipynb` and `reeval_norm_err.ipynb` currently have to be run. `tabularize.ipynb` can be used to create the tables, while other fun visualizations are available.
You can also download our relighting experiments from [here](https://drive.google.com/file/d/1CgyA1Fjis3dDDjAV3SFDJd-BGrgbOhf8/view?usp=sharing).

# Other datasets
Other dataset configurations are available in `configs/dataset`. Real world datasets are available and do work. 

Here is a [link](https://drive.google.com/file/d/131eN_Kfo-_-TOPwWjCyc02KjyOTY7bxR/view?usp=sharing) to the relighting dataset.
