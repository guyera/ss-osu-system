# BBN Intro

This package is the BBN-Symbiant-OSU Sail-On TA2 system applied to the UMD TA1 svo-classification task.

Here is a sample invocation, to be issued from the toplevel sailon-svo directory:
```bash
python main.py \
--data-root <path-to-images> \
--scg-ensemble <path-to-pretrained-SCG-ensemble> \
--pretrained-unsupervised-novelty-path <path-to-pretrained-unsupervised-module> \
--sys_results_dir <path-to-a-writeable-directory-for-temp-storage-of-results-to-be-sent-to-the-API> \
--detection_feedback
```

There is a classification_feedback argument that could be added, but our system does not currently implement it.

There is also a given_detection argument that will be supported in our final system, but is not yet supported in this version.

While our system will function if the detection_feedback argument is omitted, it will apply its initial strategy throughout, rather than demonstrating adaptation.  We are assuming that the main evaluation runs will be with detection_feedback turned on. 

If you have questions, please contact <lance.ramshaw@raytheon.com>.

The remainder of the README is taken from the UMD repo from which this was forked early in the development
of our system.

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

- [BBN Intro](#bbn-intro)
- [sailon-svo](#sailon-svo)
	- [Citations](#citations)
	- [Table of Contents](#table-of-contents)
	- [Prerequisites](#prerequisites)
	- [Data Utilities](#data-utilities)
		- [Custom Sailon Data](#custom-sailon-data)
	- [Training](#training)
	- [Testing](#testing)
	- [Baseline performance](#baseline-performance)
	- [NOTE](#note)
	- [Contact](#contact)
	- [TODO](#todo)

## Prerequisites

1. Download the repository with `git clone https://github.com/pi-umd/sailon-hoi`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)

Sample steps that can be followed if you're unable to follow the steps given above:
```bash
cd ~/sailon-svo
git clone https://github.com/fredzzhang/pocket.git
export PYTHONPATH=$PYTHONPATH:"<path to this dir>/sailon-svo/pocket"
```
3. Make sure the environment you created for Pocket is activated. You are good to go!
4. A conda environment file `svo.yml` is provided for easy installation of other prerequisites.

## Data Utilities

### Custom Sailon Data

1. Images, annotations and pretrained model is available [here](https://drive.google.com/drive/u/1/folders/18sp-dXGFyIfOH2K3ZZVfDLAna_rCEcn9)

```md5sum of model file: 70536666cc873e3b8848bb83416771bb```

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
| Triplet | Top-1 | 0.993 | 0.961      | 0.310            |
|         | Top-3 | 0.996 | 0.973      | 0.586            |
| Subject | Top-1 | 1.000 | 0.991      | 0.609            |
|         | Top-3 | 1.000 | 0.997      | 0.977            |
| Object  | Top-1 | 0.995 | 0.973      | 0.747            |
|         | Top-3 | 1.000 | 0.997      | 0.908            |

```
NOTE: We should note that model is not trained using any unknown data, so 0th index are not meaningful to use right now. They’ll be used in the next update of training data
```

## NOTE

For evaluation, we will first look for triplets in 'top_k_triplets' and if this key is empty then we will pull triplets from 'triplet_tensor' key

Hardware: single instance of NVIDIA Quandro P6000 (24G)

Inference Stats:
* Time/Iter: ~80ms
* GPU mem. footprint: ~2.6G


## Contact

If you have any questions regarding this repo, please post them in [issues](https://github.com/pi-umd/sailon-hoi/issues)
. If you ran into issues related to the code, feel free to open an issue. Alternatively, you can contact us at
anubhav@umd.edu or sonaal@umd.edu

## TODO
    Support missing subject/object boxes (-1s as input)
    Integrate with SAIL-ON API
