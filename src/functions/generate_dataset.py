from lib.mask_utils import MaskUtils
from multiprocessing import Pool
from tqdm import tqdm
import cv2, numpy as np, os, uuid

class GenerateDataset:

    def __init__(self, images, output, threads = os.cpu_count()):
        self.images = images
        self.output = output
        self.threads = threads


        self.utils = MaskUtils()

        if self.check_dirs():
            self.generate_dataset()
        else:
            print('[ ERROR ] Please check that "{}" & "{}" exist as directories!'.format(self.images, self.output))

    def check_dirs(self):
        return os.path.exists(self.images) and os.path.exists(self.output)

    def generate_dataset(self):
        os.makedirs(os.path.join(self.output, "images"), exist_ok = True)
        os.makedirs(os.path.join(self.output, "masks"), exist_ok = True)

        faces = []
        for _dir, _subdir, files in os.walk(self.images):
            for i in range(len(files)): files[i] = os.path.join(self.images, files[i])
            faces = files

        with Pool(self.threads) as p:
            list(tqdm(p.imap(self.gen_mask, faces, chunksize=self.threads), total=len(faces)))

    def gen_mask(self, face):
        id = uuid.uuid4()
        face, mask = self.utils.mask_face(face)
        if isinstance(face, np.ndarray) and isinstance(mask, np.ndarray):
            filename = "{}.png".format(id)
            cv2.imwrite(os.path.join(self.output, "images", filename), face)
            cv2.imwrite(os.path.join(self.output, "masks", filename), mask)
        return None

    
