import argparse, os
from functions.generate_dataset import GenerateDataset
from functions.training.train import set_args, train
from functions.compile_model import CompileModel

class MaskGenerator:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.gen_data = self.kwargs['generate_dataset']
        self.train = self.kwargs['train_u2net']
        self.compile = self.kwargs['compile']

        self.threads = self.kwargs['threads']
        self.images_dir = self.kwargs['images_dir']
        self.dataset_dir = self.kwargs['dataset_dir']

        self.training_bs = self.kwargs['batch_size']
        self.training_lr = self.kwargs['lr']
        self.training_lw = self.kwargs['load_weights']

        if self.gen_data: self.generate_dataset()
        elif self.train: self.train_u2net()
        elif self.compile: self.compile_model()
        else: print('[ ERROR ] Please select atleast one of the options!')

    def generate_dataset(self):
        GenerateDataset(
            self.images_dir,
            self.dataset_dir,
            self.threads
        )

    def train_u2net(self):
        set_args(
            bs = self.training_bs,
            lr = self.training_lr,
        )
        train(
            batch_size=self.training_bs,
            resume=self.training_lw
        )

    def compile_model(self):
        if not self.training_lw or not self.dataset_dir:
            print('[ERROR] Please specify the weights file to load and the dataset directory!')
        else:
            CompileModel(
                weights_file=self.training_lw,
                dataset=self.dataset_dir
            )

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--generate-dataset', default=False, action='store_true', help='Generate dataset for training U^2-NET')
    args.add_argument('--train-u2net', default=False, action='store_true', help='Train U^2-NET')
    args.add_argument('--compile', default=False, action='store_true', help='Compile the KERAS Model to be used with Tensorflow Serving. This option requires --load-weights & --dataset-dir to be set')
    args.add_argument('--threads', default=os.cpu_count(), type=int, help='Number of threads to use. Default = Num of CPU threads')
    args.add_argument('--images-dir', default='data/images', type=str, help='Images to load for generating a dataset')
    args.add_argument('--dataset-dir', default='data/dataset', type=str, help='Directory to create dataset in')

    # Training specific arguments
    args.add_argument('--batch-size', default=16, type=int, help='[TRAINING] Batch Size. Default = 16')
    args.add_argument('--lr', default=0.001, type=float, help='[TRIANING] Learning Rate. Default = 0.001')
    args.add_argument('--load-weights', default=None, type=str, help='[TRAINING] Load weights file to resume training. Default = None')

    # Compile args
    args = args.parse_args()

    MaskGenerator(**vars(args))