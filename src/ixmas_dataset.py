import os
import re
import torch
import random
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .util import download_url


class IXMASDataset(Dataset):

    def __init__(self, root, collections, transform=None, download=False, num_frames=16, verbose=False):
        self.root = root
        self.transform = transform
        self._views = 5
        self._num_frames = num_frames
        self._collections = collections
        self. is_triplets = False
        self._verbose = verbose

        if collections is None:
            raise ValueError("Collection not provided for dataset")

        if download or not self.check_exists(collections):
            self.download(collections)

        if self.check_exists(collections):
            self.clips = self.read_dataset()
            self.clip_dict = self.build_dict()
        else:
            raise ValueError("Collections available did not verify")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        if self.is_triplets:
            negative = self.get_negative(clip)

            if self._verbose:
                print("Anchor Clip: {} Collection: {} Action: {} Path: {}".format(index, clip.collection,
                                                                                  clip.label, clip.frame_paths[0][0]))
                print("Positive Clip: {} Collection: {} Action: {} Path: {}".format(index, clip.collection, clip.label,
                                                                           clip.frame_paths[1][0]))
            return clip.get_triplet(self._num_frames, negative, self.transform)
        else:
            return clip.get_single(self._num_frames, self.transform)

    def set_triplet_flag(self, flag):
        self.is_triplets = flag

    def check_exists(self, collection):
        found = 0
        dataset_path = os.path.join(self.root, "dataset/")
        calibration_path = os.path.join(dataset_path, "Calibration/")
        if os.path.exists(dataset_path) and os.path.exists(calibration_path):
            for cur_dir in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, cur_dir)) and "png" in cur_dir:
                    name = re.search("(.*)_png", cur_dir).group(1)
                    path = os.path.join(dataset_path, *[cur_dir, name + "_truth.txt"])
                    if os.path.exists(path) and name in collection:
                        found += 1

        return found == len(collection)

    def download(self, img_collections):

        cwd = os.getcwd()
        os.chdir(self.root)
        shutil.rmtree("./temp", ignore_errors=True)
        os.makedirs("./temp/truth")

        if not os.path.exists("./dataset/Calibration"):
            print("-----Retrieving Calibration-----")
            download_url("http://4drepository.inrialpes.fr/data-4d/ixmas/setup.tgz", "./temp/setup.tgz")
            tar = tarfile.open("./temp/setup.tgz")
            tar.extractall(path="./dataset/")

        print("\n-----Retrieving GroundTruth-----")
        download_url("http://4drepository.inrialpes.fr/data-4d/ixmas/data/truth.txt.tgz", "./temp/truth.txt.tgz")
        tar = tarfile.open("./temp/truth.txt.tgz")
        tar.extractall(path="./temp/truth")

        print("\n-----Retrieving Image Sets-----")
        print("Note: Depending on the number of img sets specified this may take some time.")

        with ThreadPoolExecutor(max_workers=4) as executor:
            for coll in img_collections:
                if not os.path.exists("./dataset/{}_png".format(coll)):
                    executor.submit(self.download_collection, coll)

        print("-----Completed Dataset Creation-----")
        shutil.rmtree("./temp", ignore_errors=True)
        os.chdir(cwd)

    def download_collection(self, collection):
        print("Reading {}".format(collection))
        download_name = "{}.pictures.tgz".format(collection)
        download_url("http://4drepository.inrialpes.fr/data-4d/ixmas/{}/{}".format(collection, download_name),
                     "./temp/" + download_name, progress=False)

        tar = tarfile.open("./temp/" + download_name)
        tar.extractall(path="./dataset")

        img_truth = "./temp/truth/{}_truth.txt".format(collection)
        img_truth_dest = "./dataset/{}_png/{}_truth.txt".format(collection, collection)
        shutil.copyfile(img_truth, img_truth_dest)
        print("Completing {}".format(collection))

    def read_dataset(self):
        print("-----Reading Dataset-----")

        clips = []
        dataset_path = os.path.join(self.root, "dataset/")

        paths = []
        for root, dirs, files in os.walk(dataset_path):
            for cur_dir in dirs:
                if "png" in cur_dir:
                    name = re.search("(.*)_png", cur_dir).group(1)
                    if name in self._collections:
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

                    if multi_clip is not None and multi_clip[0].frame_depth > self._num_frames:
                        clips.extend(multi_clip)

                    multi_clip = []
                    # Creates an array of clips containing all the combinations of views for the new label
                    for j in range(0, self._views):
                        for k in range(0, self._views):
                            if j != k:
                                multi_clip.append(IXMASMulticlip(path, name, current_class, j,k))

                # For each combination of views we insert a new entry into our array of clips
                for j in range(len(multi_clip)):
                    multi_clip[j].add_frame(path, i)

        print("Successfully created {} samples from {} collections and {} actions.".format(len(clips),
                                                                                          len(self._collections), 13))
        print("-----Completed Dataset-----")

        return clips

    def build_dict(self):

        collection_dict = {collection: {cam: {} for cam in range(self._views)} for collection in self._collections}
        for i in range(len(self.clips)):
            clip = self.clips[i]
            action = clip.label
            collection = clip.collection
            cam = clip.cam1
            if action not in collection_dict[collection][cam]:
                collection_dict[collection][cam][action] = [i]
            else:
                collection_dict[collection][cam][action].append(i)

        return collection_dict

    def get_negative(self, clip):
        action = clip.label
        cam = clip.cam1
        collection = self.clip_dict[clip.collection][cam]

        search_action = action
        while search_action == action:
            search_action = random.choice(list(collection.keys()))

        found_index = random.choice(collection[search_action])
        negative_clip = self.clips[found_index]

        if self._verbose:
            print("Negative Clip: {} Collection: {} Action: {} Path: {}".format(found_index, negative_clip.collection,
                                                                                negative_clip.label, negative_clip.
                                                                                frame_paths[0][0]))
        return negative_clip.get_negative(self._num_frames, self.transform)


class IXMASMulticlip:

    def __init__(self, root, collection, label, cam1, cam2):
        self._root = root
        self.frame_paths = [[], []]
        self.collection = collection
        self.label = label
        self.cam1 = cam1
        self.cam2 = cam2

    @property
    def frame_depth(self):
        return len(self.frame_paths[0])

    def add_frame(self, path, frame):
        path1 = os.path.join(path, *["cam" + str(self.cam1), "img{}.png".format(format(frame, '04d'))])
        path2 = os.path.join(path, *["cam" + str(self.cam2), "img{}.png".format(format(frame, '04d'))])
        self.frame_paths[0].append(path1)
        self.frame_paths[1].append(path2)

    def get_start_index(self, num_frames):
        max_bound = self.frame_depth - num_frames
        anchor_index = random.randint(0, max_bound)
        return anchor_index

    def get_triplet(self, num_frames, negative, transform=None):
        index = self.get_start_index(num_frames)
        anchor = self.get_subclip(0, num_frames, index, transform)
        positive = self.get_subclip(1, num_frames, index, transform)

        return torch.cat((anchor, positive, negative), 0)

    def get_negative(self, num_frames, transform=None):
        index = self.get_start_index(num_frames)
        negative = self.get_subclip(0, num_frames, index, transform)
        return negative

    def get_single(self, num_frames, transform=None):
        index = self.get_start_index(num_frames)
        cam = random.randint(0, 1)
        return self.get_subclip(cam, num_frames, index, transform)

    def get_subclip(self, view, num_frames, frame_start, transform=None):
        frame_set = self.frame_paths[view][frame_start:frame_start + num_frames]
        return self.load_frames(frame_set, transform)

    def load_frames(self, frame_set, transform=None):

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        frames = None
        for frame in frame_set:
            img = transform(Image.open(frame))
            img = img.unsqueeze(0)
            if frames is None:
                frames = img
            else:
                frames = torch.cat((frames, img))

        frames = torch.transpose(frames, 0, 1)
        frames = frames.unsqueeze(0)

        return frames
