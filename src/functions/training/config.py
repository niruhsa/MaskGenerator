import pathlib

# Model
resume = None
weight_dir = pathlib.Path('data/weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (256, 256, 3)
default_out_shape = (256, 256, 1)

# Training
batch_size = 12
epochs = 1000000
learning_rate = 0.001
save_interval = 250

# Dataset 
root_data_dir = pathlib.Path('data')
dataset_dir = root_data_dir.joinpath('dataset')
image_dir = dataset_dir.joinpath('images')
mask_dir = dataset_dir.joinpath('masks')

# Evaluation
output_dir = pathlib.Path('data/weights')
