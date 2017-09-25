# Generate-Face-Image-by-GAN
Generate Face Image by DCGAN

mnist.py: a simple GAN model training on mnist data

download.py: download data from the Internet
	python download.py celebA
	
operations.py: some useful functions defined here

test.py: the main program


## Usage

First, download dataset with:

    $ python download.py mnist celebA

To train a model with downloaded dataset:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --train
    $ python main.py --dataset celebA --input_height=108 --train --crop
    $ python main.py --dataset Webface --input_height=256 --crop True --train True --train_size 100

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28
    $ python main.py --dataset celebA --input_height=108 --crop
    $ python main.py --dataset Webface --input_height=256 --crop True
	
Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train
