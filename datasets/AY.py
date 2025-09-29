# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import random

class AY20(BaseImageDataset):
    dataset_dir = 'Anyang-reid-format/2020/Night/'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(AY20, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'IR')
        self.gallery_dir = osp.join(self.dataset_dir, 'IR')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = [] # self._process_dir([], relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        query = self._select_random_one_per_pidcam(query, seed=0)

        q_paths = {p for (p, _, _, _) in query}
        gallery = [x for x in gallery if x[0] not in q_paths]

        if verbose:
            print("=> AY20 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _select_random_one_per_pidcam(self, dataset, seed=0):
        by_group = defaultdict(list)
        for item in dataset:
            _, pid, camid, _ = item
            by_group[(pid, camid)].append(item)

        rnd = random.Random(seed)

        keys = sorted(by_group.keys(), key=lambda x: (x[0], x[1]))
        picked = [rnd.choice(by_group[key]) for key in keys]

        picked.sort(key=lambda x: (x[1], x[2], x[0]))

        return picked


class AY21(BaseImageDataset):
    dataset_dir = 'Anyang-reid-format/'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(AY21, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, '2021')
        self.gallery_dir = osp.join(self.dataset_dir, '2021')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = [] # self._process_dir([], relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        query = self._select_random_one_per_pidcam(query, seed=0)

        q_paths = {p for (p, _, _, _) in query}
        gallery = [x for x in gallery if x[0] not in q_paths]

        if verbose:
            print("=> AY21 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _select_random_one_per_pidcam(self, dataset, seed=0):
        by_group = defaultdict(list)
        for item in dataset:
            _, pid, camid, _ = item
            by_group[(pid, camid)].append(item)

        rnd = random.Random(seed)

        keys = sorted(by_group.keys(), key=lambda x: (x[0], x[1]))
        picked = [rnd.choice(by_group[key]) for key in keys]

        picked.sort(key=lambda x: (x[1], x[2], x[0]))

        return picked