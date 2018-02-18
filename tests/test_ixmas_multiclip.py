from unittest import TestCase
import os
from src.ixmas_dataset import IXMASDataset


class TestIXMASMulticlip(TestCase):

    def __init__(self):
        self.dataset = IXMASDataset(os.path.abspath("../"), ["julien1"])
        if self.dataset.check_exists(["julien1"]):
            self.multiclip = self.dataset[0]

            self.test_frame_depth()
            self.test_get_triplet_indexes()

        else:
            self.fail()

    def test_frame_depth(self):
        if self.multiclip.frame_depth != 47:
            self.fail()

    def test_get_triplet_indexes(self):
        num_frames = 16

        for i in range(10):
            indexes = self.multiclip.get_triplet_indexes(num_frames)

            if indexes[0] != indexes[1]:
                self.fail()

            if abs(indexes[0] - indexes[2]) < num_frames:
                self.fail()

            if indexes[0] > self.multiclip.frame_depth or indexes[2] > self.multiclip.frame_depth:
                self.fail()

            if indexes[0] < 0 or indexes[2] < 0:
                self.fail()

            if indexes[0] + num_frames > self.multiclip.frame_depth or \
                    indexes[2] + num_frames > self.multiclip.frame_depth:
                self.fail()


    def test_get_triplet(self):
        self.fail()

    def test_get_subclip(self):
        self.fail()

    def test_load_frames(self):
        self.fail()
