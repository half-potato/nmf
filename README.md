# Neural Microfacet Fields for Inverse Rendering
More details can be found at the project page [here](https://half-potato.gitlab.io/posts/nmf/).
# Installation
A conda virtual environment is recommended.
```
pip install -r requirements.txt
```
Dataset dir should contain a folder named `nerf_synthetic` with various datasets in the `blender` configuration.
```
python train.py -m expname=v38_noupsample model=brdf_tcnn dataset=ficus,drums,ship,teapot vis_every=5000 datadir={dataset dir}
```
Experiment configurations are done using hydra, which controls the initialization parameters for all of the modules. Look in `configs/model` to 
see what options are available. Setting the BRDF activation would look like adding this:
```
model.arch.model.brdf.activation="sigmoid"
```
to the command line argument.

# Recreating Experiments
Note that something is currently wrong with computation of metrics in the current code and the scripts `reval_lpips.ipynb` and `reeval_norm_err.ipynb` currently have to be run. `tabularize.ipynb` can be used to create the tables, while other fun visualizations are available.

# Other datasets
Other dataset configurations are available in `configs/dataset`. Real world datasets are available and do work. 
