#!/usr/bin/python3

import numpy as np
import os
import torch
from ..home import get_project_base
from yacs.config import CfgNode
from dataclasses import dataclass
from .analysis import parse_label

def actionset_label(transcript, num_classes):
    action = torch.zeros(num_classes).float()
    for i in range(num_classes):
        if i in transcript:
            action[i] = 1
    
    return action

def vname2recipe(dataset: str, vname: str, recipe2index: dict) -> int:
    dataset = dataset.lower()
    if dataset == "breakfast":
        recipe = recipe2index[vname.split("_")[3]]
    elif dataset == "crosstask":
        recipe = recipe2index[vname.split('.')[0]]
    elif dataset == 'mc2':
        recipe = recipe2index[vname.split('-')[1][1:]] # integer to integer
    else:
        raise ValueError(dataset)

    return recipe

def create_recipe_dict(dataset):
    recipe_dict = {}
    for vname in dataset.get_vnames():
        recipe = dataset.recipe_label[vname]
        if recipe not in recipe_dict:
            recipe_dict[recipe] = []
        recipe_dict[recipe].append(vname)
    return recipe_dict


class Dataset(object):
    """
    self.features[video]: the feature array of the given video (dimension x frames)
    self.transcrip[video]: the transcript (as label indices) for each video
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, dataset_name, base_path, feature_dir, video_list, label2index, recipe2index,
                        transpose=False, feature_type='.npy'):
        self.video_list = video_list
        self.dataset = dataset_name
        self.features = dict()
        self.transcript = dict()
        self.gt_label = dict()
        self.recipe_label = dict()

        self.base_path = base_path
        self.feature_dir = feature_dir
        # read features for each video
        base_path = base_path.rstrip('/')
        for video in video_list:
            # recipe_labe
            self.recipe_label[video] = vname2recipe(dataset_name, video, recipe2index)
            # video features
            if feature_dir is not None: # if None, skip feature loading
                self.features[video] = \
                    self._load_feature(feature_dir, video, feature_type, transpose) # should be D x T
            # gt_label
            with open(base_path + '/groundTruth/' + video + '.txt') as f:
                self.gt_label[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
            # transcript
            segments = parse_label(self.gt_label[video])
            self.transcript[video] = [ s.action for s in segments ]

        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0] if feature_dir is not None else -1
        self.n_classes = len(label2index)

    def _load_feature(self, feature_dir, video, feature_type, transpose):
        file_name = os.path.join(feature_dir, video+feature_type)
        if feature_type == '.npy':
            feature = np.load(file_name)
        elif feature_type == '.npz':
            feature = np.load(file_name)
            feature = feature['arr_0']

        if transpose:
            feature = feature.T
        if feature.dtype != np.float32:
            feature = feature.astype(np.float32) 
        
        return feature


    def __str__(self):
        string = "< Dataset %d videos, %d feat-size, %d classes, feature-dir: %s >"
        string = string % (len(self.video_list), self.input_dimension, self.n_classes, self.feature_dir)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list

    def __getitem__(self, video):
        return self.features[video], self.recipe_label[video], self.transcript[video], self.gt_label[video]

    def __len__(self):
        return len(self.gt_label)

class DataLoader():

    def __init__(self, dataset, batch_size, shuffle=False):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video/self.batch_size))

        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
            self.index = 0
            raise StopIteration

        else:
            video_idx = self.selector[self.index : self.index+self.batch_size]
            if len(video_idx) < self.batch_size:
                video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence, batch_action_label = [], []
            batch_trans = []
            batch_gt_label = []
            for vfname in videos:
                sequence, recipe, trans, gt_label = self.dataset[vfname]
                sequence = np.expand_dims(sequence, 0) # should be 1, D, T
                action = actionset_label(trans, self.dataset.n_classes)
                
                batch_sequence.append(sequence)
                batch_action_label.append(action)

                batch_trans.append(torch.LongTensor(trans))
                batch_gt_label.append(torch.LongTensor(gt_label))

            batch_sequence = [ torch.from_numpy(s) for s in batch_sequence ]
            tensor_action_label = torch.stack(batch_action_label, dim=0)

            #WARN batch_gt_label is a list
            labels = {
                "action_label": tensor_action_label,
                "transcript"  : batch_trans,
                "gt_label"    : batch_gt_label,
            }

            return videos, batch_sequence, labels 


class DataLoader_GroupSample():

    def __init__(self, dataset, batch_size, shuffle=False):
        assert shuffle

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video/self.batch_size))

        self.group_dict = create_recipe_dict(dataset)
        self.index = 0

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video: # mimic end of epoch
            self.index = 0
            raise StopIteration

        # choose a recipe
        groups = list(self.group_dict)
        group = np.random.choice(groups, size=1).item()
        vnames = self.group_dict[group]
        num_video = min(self.batch_size, len(vnames))

        # choose video
        vnames = np.random.choice(vnames, size=num_video, replace=False)

        # load video
        batch_sequence, batch_action_label = [], []
        batch_trans = []
        batch_gt_label = []
        for vfname in vnames:
            sequence, recipe, trans, gt_label = self.dataset[vfname]
            sequence = np.expand_dims(sequence, 0) # should be 1, D, T

            action = actionset_label(trans, self.dataset.n_classes)
            
            batch_sequence.append(sequence)
            batch_action_label.append(action)

            batch_trans.append(torch.LongTensor(trans))
            batch_gt_label.append(torch.LongTensor(gt_label))

        #WARN batch_sequence is a list
        batch_sequence = [ torch.from_numpy(s) for s in batch_sequence ]
        tensor_action_label = torch.stack(batch_action_label, dim=0)

        labels = {
            "action_label": tensor_action_label,
            "transcript"  : batch_trans,
            "gt_label"    : batch_gt_label,
        }

        self.index += num_video

        return vnames, batch_sequence, labels 



    
@dataclass
class DatasetInfo():
    map_fname: str
    recipe_map_fname: str
    dataset_dir: str
    feature_dir: str
    train_split_fname: str
    test_split_fname: str
    transpose: bool
    feature_type: str

def get_dataset_paths(cfg: CfgNode) -> DatasetInfo :
    BASE = get_project_base()
    transpose = False
    feature_type = '.npy'
    if cfg.name == "breakfast":
        map_fname = BASE + 'dataset/Breakfast/mapping.txt'
        recipe_map_fname = BASE + 'dataset/Breakfast/recipe_mapping.txt'
        dataset_dir = BASE + 'dataset/Breakfast/'
        feature_dir = BASE + 'dataset/Breakfast/features/'
        transpose = False

        train_split_fname = BASE + 'dataset/Breakfast/split%d.train' % cfg.split
        test_split_fname = BASE + 'dataset/Breakfast/split%d.test' % cfg.split

    elif cfg.name == "crosstask":
        map_fname = BASE + 'dataset/CrossTask/mapping.txt'
        recipe_map_fname = BASE + 'dataset/CrossTask/recipe_mapping.txt'
        dataset_dir = BASE + 'dataset/CrossTask/'
        feature_dir = BASE + 'dataset/CrossTask/features/'
        transpose = False
        train_split_fname = BASE + 'dataset/CrossTask/split%d.train' % cfg.split
        test_split_fname = BASE + 'dataset/CrossTask/split%d.test' % cfg.split

    elif cfg.name == "mc2":
        assert cfg.split == 1
        feature_dir = BASE + 'dataset/MC2/features/'
        transpose = True
        feature_type = '.npz'
        map_fname = BASE + 'dataset/MC2/mapping.txt'
        recipe_map_fname = BASE + 'dataset/MC2/recipe_mapping.txt'
        dataset_dir = BASE + 'dataset/MC2/'
        train_split_fname = BASE + 'dataset/MC2/split%d.train' % cfg.split
        test_split_fname = BASE + 'dataset/MC2/split%d.test' % cfg.split

    info = DatasetInfo(map_fname, recipe_map_fname, dataset_dir, feature_dir, 
                        train_split_fname, test_split_fname, 
                        transpose, feature_type)

    return info



def load_action_mapping(map_fname):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    return label2index, index2label