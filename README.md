# Panorama Stitcher
This rep can stitch multi panorama images. It contains several deep and image-based analysis to do so.



## Available Methods and Preliminaries
This library offers several panorama stitching methods based on different techniques.

General options can be selected before applying the stitching methods. These options are:
- You should add the path for the image directory with the help of `-d` or `--data_path`.
- One can resize the image with `-r` or `--resize_shape`. (Default value is `None` which uses the original size of the images.)
- Define the result path with `--result_path` or `-s` (Default directory is `./`).
- Select the verbose value for logging for example as `-v` depending on what kind of logs you want to see.

The available methods are as:
- [Simple Opencv Stitcher](#simple-opencv-stitcher)
- [Detailed Stitching/Opencv Stitcher](#detailed-stitching/opencv-stitcher)
- [Kornia Stitcher](#kornia-stitcher)
- [Keypoint Stitcher](#keypoint-stitcher)
- [Sequential Stitcher](#sequential-stitcher)

### Simple OpenCV Stitcher
This method mainly uses stitcher class from opencv to create the panorama images from multi images. It is one of the fastest and applicable methods in case of multi-image stitching.
Only one option should be set for this method as:
- Set `--stitcher_type` as "scan" if the images are scan or "panorama" if images are panorama images.

A simple code to stitch boat test images is:
```shell
panaroma_stitcher -vv -d ./test_data/boat opencv-simple --stitcher_type panorama
```
This method is recommended than other methods as it is fast, and it can stitch multi high resolution images properly. As an example:
<p align="center">
    <img width="1000" src="./results/boat_simple_stitcher.png" alt="Simple Stitcher">
</p>

### Detailed Stitching/Opencv Stitcher
This method is a more detailed version of the above simple stitcher methods based on opencv/stitching libraries. It is quite fast and have good accuracy
specifically in stitching multi images. It is important to know that in some cases it is needed to modify the methods parameters to get accurate results.
Some of the options that can be set in this method are:
- `--detect_method` defines the key point detection methods and can be chosen from "sift", "orb", "brisk", or "akaze".
- `--match_type` defines the matching method and can be selected as "affine" or "homography".
- `--num_feat` defines the number of features in the detector.
- `--device` whether to use the gpu or cpu as processing unit.
- `--conf_thr` defines the threshold for finding key points.
- `--cam_est` defines the camera estimator to be "affine" or "homography".
- `--cam_adj` defines the camera adjustor to be "ray", "reproj", "affine", or "no".

Some examples of running this code for test images are:
```shell
panaroma_stitcher -vv -d ./test_data/boat detailed-stitcher --detect_method sift --match_type homography --num_feat 500 --device cpu --conf_thr 0.05 --cam_est homography --cam_adj ray
panaroma_stitcher -vv -d ./test_data/map detailed-stitcher --detect_method brisk --match_type homography --num_feat 500 --device cpu --conf_thr 0.05 --cam_est homography --cam_adj ray
panaroma_stitcher -vv -d ./test_data/castle detailed-stitcher --detect_method orb --match_type homography --num_feat 500 --device cpu --conf_thr 0.05 --cam_est homography --cam_adj ray
panaroma_stitcher -vv -d ./test_data/newspaper detailed-stitcher --detect_method brisk --match_type homography --num_feat 500 --device cpu --conf_thr 0.05 --cam_est homography --cam_adj ray
```
As an example of:
<p align="center">
    <img width="1000" src="./results/map_detailed_stitcher.png" alt="Detailed Stitcher">
</p>


### Kornia Stitcher
This method is based on `kornia` library, and three feature matcher as `LOFTR` deep feature matcher, `GFTTAffNetHardNet`,
and `KeyNetAffNetHardNet` matcher. The accuracy of this method is good specifically for stitching two images. However, since it uses a deep feature extractor and matcher
it requires large memory and is a bit slow. It uses a GPU automatically in cases where CUDA is installed.
Some options are:
- `--method` is used to define the matching method as "loftr", "local", "keynote".
- `--loftr_model` should be selected as "indoor" or "outdoor" depending on where the images are taken is someone uses "loftr" matcher.
- `--features` is the number of features in "local" and "keynote" methods.
- `--matcher` defines the matching algorithms in "local" or "keynote" methods. It can be "snn", "nn", "mnn", or "smnn".

Some examples of using these methods:
```shell
panaroma_stitcher -vv -d ./test_data/mountain kornia --method loftr --loftr_model outdoor
panaroma_stitcher -vv -d ./test_data/river kornia --method local --features 100 --matcher smnn
```
As an example of:
<p align="center">
    <img src="./results/mountain_kornia_stitcher.png" alt="Kornia Stitcher">
</p>

### Sequential Stitcher
This is a simple, fast, and accurate and sequential stitcher images that tries to find the similarity between pairs of images and then locate the images in the final stitched image properly.
It is good to know that the current code is written in a such a way that the input images should be taken from left to right.
If it is not the case then a simple modification should be done based on how the images are taken (for example if they are taken from right to left.)
Some options for this method are:
- `--matching_method` to be selected as "bf" or "flann".
- `--detector_method` to be selected as "sift", "orb", or "brisk".
- `--number_feature` can affect the performance significantly in some cases.
- `--final_shape` is the final image size.

Some examples of using this method:
```shell
panaroma_stitcher -vv -d ./test_data/boat sequential-stitcher --matching_method bf --detector_method sift --number_feature 500 --final_shape 3000 18000
panaroma_stitcher -vv -d ./test_data/river sequential-stitcher --matching_method bf --detector_method sift --number_feature 500 --final_shape 1000 3000
panaroma_stitcher -vv -d ./test_data/mountain sequential-stitcher --matching_method bf --detector_method sift --number_feature 500 --final_shape 300 800
```
As an example:
<p align="center">
    <img width="1000" src="./results/river_sequential_stitcher.png" alt="Kornia Stitcher">
</p>

### Keypoint Stitcher
This is a simple stitcher that tries to stitch a pair of images from a folder recursively. It performs well in some cases where the other methods do not work well.
However, it might be a bit slow if the number of features in the detector are large. Some options for this method are:
- `--matching_method` to be selected as "bf" or "flann".
- `--detector_method` to be selected as "sift", "orb", or "brisk".
- `--number_feature` can affect the performance significantly in some cases.

Some examples of using this method:
```shell
panaroma_stitcher -vv -d ./test_data/mountain keypoint-stitcher --matching_method bf --detector_method sift --number_feature 500
panaroma_stitcher -vv -d ./test_data/river keypoint-stitcher --matching_method bf --detector_method sift --number_feature 500
```
As an example:
<p align="center">
    <img src="./results/castle_keypoint_stitcher.png" alt="Kornia Stitcher">
</p>

## How to Develop
Do the following only once after creating your project:
- Init the git repo with `git init`.
- Add files with `git add .`.
- Then `git commit -m 'initialize the project'`.
- Add remote url with `git remote add origin REPO_URL`.
- Then `git branch -M master`.
- `git push origin main`.
Then create a branch with `git checkout -b BRANCH_NAME` for further developments.
- Install poetry if you do not have it in your system from [here](https://python-poetry.org/docs/#installing-with-pipx).
- Create a virtual env preferably with virtualenv wrapper and `mkvirtualenv -p $(which python3.10) ENVNAME`.
- Then `git add poetry.lock`.
- Then `pre-commit install`.
- For applying changes use `pre-commit run --all-files`.

## Docker Container
To run the docker with ssh, do the following first and then based on your need select ,test, development, or production containers:
```shell
export DOCKER_BUILDKIT=1
export DOCKER_SSHAGENT="-v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK -e SSH_AUTH_SOCK"
```
### Test Container
This container is used for testing purposes while it runs the test
```shell
docker build --progress plain --ssh default --target test -t panaroma_docker:test .
docker run -it --rm -v "$(pwd):/app" $(echo $DOCKER_SSHAGENT) panaroma_docker:test
```

### Development Container
This container can be used for development purposes:
```shell
docker build --progress plain --ssh default --target development -t panaroma_docker:development .
docker run -it --rm -v "$(pwd):/app" -v /tmp:/tmp $(echo $DOCKER_SSHAGENT) panaroma_docker:development
```

### Production Container
This container can be used for production purposes:
```shell
docker build --progress plain --ssh default --target production -t panaroma_docker:production .
docker run -it --rm -v "$(pwd):/app" -v /tmp:/tmp $(echo $DOCKER_SSHAGENT) panaroma_docker:development panaroma_stitcher -vv -d ./test_data/boat opencv-simple --stitcher_type panorama
```

## Hugging Face