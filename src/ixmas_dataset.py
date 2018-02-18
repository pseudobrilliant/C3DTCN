import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import random
from torch.utils.data import Dataset, DataLoader
from urllib.request import urlretrieve


class IXMASDataset(Dataset):

    def __init__(self, root, transform=None, download=False):
        self.root = root
        self.transform = transform
        self._views = 5

        if download or not self.check_exists():
            self.download()

        if self.check_exists():
            self.clips = self.read_dataset()
        else:
            print("Error: Unable to find dataset")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        return self.clips[item]

    def check_exists(self):
        found = False
        dataset_path = os.path.join(self.root, "dataset/")
        calibration_path = os.path.join(dataset_path, "Calibration/")
        if os.path.exists(dataset_path) and os.path.exists(calibration_path):
            for cur_dir in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, cur_dir)) and "png" in cur_dir:
                    name = re.search("(.*)_png", cur_dir).group(1)
                    path = os.path.join(dataset_path, *[cur_dir, name + "_truth.txt"])
                    if os.path.exists(path):
                        found = True
                    else:
                        return False
        return found

    def download(self):
        import shutil
        import tarfile

        cwd = os.getcwd()
        os.chdir(self.root)
        shutil.rmtree("./temp", ignore_errors=True)
        shutil.rmtree("./dataset", ignore_errors=True)
        os.makedirs("./temp/truth")
        os.makedirs('./dataset')

        print("-----Retrieving Calibration-----")
        urlretrieve("http://4drepository.inrialpes.fr/data-4d/ixmas/setup.tgz", "./temp/setup.tgz")
        tar = tarfile.open("./temp/setup.tgz")
        tar.extractall(path="./dataset/")

        print("-----Retrieving GroundTruth-----")
        urlretrieve("http://4drepository.inrialpes.fr/data-4d/ixmas/data/truth.txt.tgz", "./temp/truth.txt.tgz")
        tar = tarfile.open("./temp/truth.txt.tgz")
        tar.extractall(path="./temp/truth")

        print("-----Retrieving Image Sets-----")
        print("Note: Depending on the number of img sets specified this may take some time.")
        # img_paths = ["julien1", "alba1", "alba2", "alba3", "amel1", "amel2", "amel3"]
        img_paths = ["julien1"]

        for i in img_paths:
            print("Reading {}".format(i))
            download_name = "{}.pictures.tgz".format(i)

            urlretrieve("http://4drepository.inrialpes.fr/data-4d/ixmas/{}/{}".format(i, download_name),
                        "./temp/" + download_name)
            tar = tarfile.open("./temp/" + download_name)
            tar.extractall(path="./dataset")

            img_truth = "./temp/truth/{}_truth.txt".format(i)
            img_truth_dest = "./dataset/{}_png/{}_truth.txt".format(i, i)
            shutil.copyfile(img_truth, img_truth_dest)
            print("Completing {}".format(i))

        print("-----Completed Dataset Creation-----")
        shutil.rmtree("./temp", ignore_errors=True)
        os.chdir(cwd)

    def read_dataset(self):
        if self.check_exists():
            clips = []
            dataset_path = os.path.join(self.root, "dataset/")

            paths = []
            for root, dirs, files in os.walk(dataset_path):
                for cur_dir in dirs:
                    if "png" in cur_dir:
                        paths.append(cur_dir)

            for cur_dir in paths:
                name = re.search("(.*)_png", cur_dir).group(1)
                path = os.path.join(dataset_path, cur_dir)
                truth = os.path.join(path, name + "_truth.txt")
                with open(truth) as fp:
                    truth_values = fp.read().split("\n")

                current_class = -1
                multi_clip = None
                for i in range(len(truth_values)):
                    if truth_values[i] != current_class:
                        current_class = truth_values[i]

                        if multi_clip is not None:
                            clips.extend(multi_clip)

                        # Creates an array of clips containing all the combinations of views for the new label
                        multi_clip = [ixmas_mvclip(path, current_class) for j in
                                      range(self._views * (self._views - 1))]

                    # For each combination of views we insert a new entry into our array of clips
                    multi_index = 0
                    for j in range(5):
                        for k in range(5):
                            if k != j:
                                multi_clip[multi_index].add_frame(0, os.path.join(path, *["cam" + str(j),
                                                                                          "img{}.png".format(
                                                                                              format(i, '04d'))]))
                                multi_clip[multi_index].add_frame(1, os.path.join(path, *["cam" + str(k),
                                                                                          "img{}.png".format(
                                                                                              format(i, '04d'))]))
                                multi_index += 1

        return clips


class ixmas_mvclip:

    def __init__(self, root, label):
        self._root = root
        self._frame_paths = [[], []]
        self._label = label

    @property
    def frame_depth(self):
        return len(self._frame_paths[0])

    def add_frame(self, camera, frame):
        self._frame_paths[camera].append(frame)

    def get_label(self):
        return self._label

    def get_triplet(self, num_frames):
        max_bound = self.frame_depth - num_frames
        anchor_index = random.randint(0, max_bound)
        anchor_end = anchor_index + num_frames

        low_bound = 0 if (self.frame_depth - anchor_end) / num_frames < 1 else anchor_end
        max_bound = (anchor_index - num_frames) if low_bound == 0 else max_bound
        negative_index = random.randint(0, max_bound)

        anchor = self.get_subclip(0, num_frames, anchor_index)
        positive = self.get_subclip(1, num_frames, anchor_index)
        negative = self.get_subclip(0, num_frames, negative_index)

        return np.concatenate((anchor, positive, negative))

    def get_subclip(self, view, num_frames, frame_start):
        frame_set = self._frame_paths[view][frame_start:frame_start + num_frames]
        return self.load_frames(frame_set)

    def load_frames(self, frame_set):
        frames = np.array([resize(io.imread(frame), output_shape=(291, 390), preserve_range=True) for frame in frame_set])
        frames = frames.transpose(3, 0, 1, 2)
        frames = np.expand_dims(frames, axis=0)
        frames = np.float32(frames)
        return torch.from_numpy(frames)
