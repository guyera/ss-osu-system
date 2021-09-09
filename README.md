DISCLAIMER: Most of these instructions are copied from repos mentioned in the `DISCLAIMER` file. We are declaring this
explicitly here, again, to avoid any claims of plagiarism.

# sailon-hoi

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
- [Testing](#testing)
- [Training](#training)
- [Contact](#contact)

## Prerequisites

1. Download the repository with `git clone https://github.com/pi-umd/sailon-hoi`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
3. Make sure the environment you created for Pocket is activated. You are good to go!

## Data Utilities

The [HICO-DET](https://github.com/pi-umd/hicodet) and [V-COCO](https://github.com/pi-umd/vcoco) repos have been
incorporated as submodules for convenience. To download relevant data utilities, run the following commands.

```bash
cd /path/to/sailon-hoi
git submodule init
git submodule update
```

### HICO-DET

1. Download the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/sailon-hoi/hicodet
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link
    ```bash
    cd /path/to/sailon-hoi/hicodet
    ln -s /path/to/hico_20160224_det ./hico_20160224_det
    ```
2. Run a Faster R-CNN pre-trained on MS COCO to generate detections

```bash
cd /path/to/sailon-hoi/hicodet/detections
python preprocessing.py --partition train2015
python preprocessing.py --partition test2015
```

```
The labels in these detections will be used as GT for training the box classification heads. If you want to use GT
boxes and labels for this purpose, please edit the output files of these scrips accordingly.
```

3. Generate ground truth detections (optional)

```bash
cd /path/to/spatially-conditioned-graphs/hicodet/detections
python generate_gt_detections.py --partition test2015 
python generate_gt_detections.py --partition train2015 
```

4. Download fine-tuned detections (optional)

```bash
cd /path/to/spatially-conditioned-graphs/download
bash download_finetuned_detections.sh
```

5. Download the *data* folder needed for
   IDN [here](https://drive.google.com/file/d/1iHVHrIsJTT97Bmlb1yuLv76C-07YljeP/view?usp=sharing).

To attempt fine-tuning yourself, refer to
the [instructions](https://github.com/pi-umd/hicodet/tree/main/detections#fine-tune-the-detector-on-hico-det) in
the [HICO-DET repository](https://github.com/pi-umd/hicodet). The checkpoint of original author's fine-tuned detector
can be found [here](https://drive.google.com/file/d/11lS2BQ_In-22Q-SRTRjRQaSLg9nSim9h/view?usp=sharing).

## Testing

```bash
cd /path/to/sailon-hoi
python test.py --net scg --world-size 8 &>log &
```

Specify the number of GPUs to use with the argument `--world-size` and specify the model you want to use with `--net`

## Training

```bash
cd /path/to/sailon-hoi
python train.py --net scg --world-size 8 &>log &
```

Specify the number of GPUs to use with the argument `--world-size` and specify the model you want to use with `--net`

## Contact

If you have any questions regarding this repo, please post them in [issues](https://github.com/pi-umd/sailon-hoi/issues)
. If you ran into issues related to the code, feel free to open an issue. Alternatively, you can contact us at
anubhav@umd.edu or vinoj@umd.edu


Release information for code merged on 9/9/2021

    Separate model folders for model files
    Separate box_coords, box_scores for subjects and objects
    Expose verb features
    Remove reference to pair and rename as verb
    Restructure the output dictionary
    Separate classification heads for subject and object
        NOTE: Will work only with same backbone and same no. of classes for now.
    Data loader for custom datasets

