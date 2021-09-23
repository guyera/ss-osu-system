DISCLAIMER: Most of these instructions are copied from repos mentioned in the `DISCLAIMER` file. We are declaring this
explicitly here, again, to avoid any claims of plagiarism.

# sailon-svo

This is the repo for the SAIL-ON SVO. Spatially Conditioned Graphs (SCG) is the baseline. 

## Citations

```bibtex
@article{zhang2020,
	author = {Frederic Z. Zhang and Dylan Campbell and Stephen Gould},
	title = {Spatially Conditioned Graphs for Detecting Human-Object Interactions},
	journal = {arXiv preprint arXiv:2012.06060},
	year = {2020}
}
```

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Utilities](#data-utilities)
    * [HICO-DET](#hico-det)
    * [SAIL-ON Custom Data](#custom-sailon-data)
- [Testing](#testing)
- [Training](#training)
- [Contact](#contact)

## Prerequisites

1. Download the repository with `git clone https://github.com/pi-umd/sailon-hoi`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
3. Make sure the environment you created for Pocket is activated. You are good to go!
4. A conda environment file `svo.yml` is provided for easy installation of other prerequisites.

## Data Utilities

The [HICO-DET](https://github.com/pi-umd/hicodet) and [V-COCO](https://github.com/pi-umd/vcoco) repos have been
incorporated as submodules for convenience. To download relevant data utilities, run the following commands.

```bash
cd /path/to/sailon-hoi
git submodule init
git submodule update
```
### Custom Sailon Data

1. Images, annotations and pretrained model is available [here](https://drive.google.com/drive/u/1/folders/18sp-dXGFyIfOH2K3ZZVfDLAna_rCEcn9)


## Training

```bash
cd /path/to/sailon-hoi
python train.py --net scg --world-size 8 &>log &
```

Specify the number of GPUs to use with the argument `--world-size` and specify the model you want to use with `--net`

## Testing

```bash
cd /path/to/sailon-hoi
python test.py --net scg --world-size 8 &>log &
```

Specify the number of GPUs to use with the argument `--world-size` and specify the model you want to use with `--net`

## Baseline performance

Top-k performance of the SCG model on the custom SAIL-ON data

|         |       | Train | Validation | Novel Validation |
|---------|-------|-------|------------|------------------|
| Triplet | Top-1 | 0.992 | 0.955      | 0.540            |
|         | Top-3 | 0.995 | 0.970      | 0.690            |
| Subject | Top-1 | 0.994 | 0.994      | 0.793            |
|         | Top-3 | 1.000 | 1.000      | 0.989            |
| Object  | Top-1 | 1.000 | 0.964      | 0.747            |
|         | Top-3 | 1.000 | 0.997      | 0.920            |

Hardware: single instance of NVIDIA Quandro P6000 (24G)

Inference Stats:
* Time/Iter: ~80ms
* GPU mem. footprint: ~2.6G


## Contact

If you have any questions regarding this repo, please post them in [issues](https://github.com/pi-umd/sailon-hoi/issues)
. If you ran into issues related to the code, feel free to open an issue. Alternatively, you can contact us at
anubhav@umd.edu or vinoj@umd.edu

## Release information for code merged on 9/15/2021

    Separate model folders for model files
    Separate box_coords, box_scores for subjects and objects
    Expose verb features
    Remove reference to pair and rename as verb
    Restructure the output dictionary
    Separate classification heads for subject and object
        NOTE: Will work only with same backbone and same no. of classes for now.
    Data loader for custom dataset
    3D Matrix for SVO probability calculation
    README updated for custom data
    Environment file included for easy installation


## TODO
    Support “unknown”/“null” classes. 
    Support missing subject/object boxes (-1s as input). 
    Support for different number of subjects and objects. 
    Integrate with SAIL-ON API.
    Remove HICO-DET and V-COCO sub modules