#!/usr/bin/python3

import numpy as np
from .utils.dataset import Dataset, DataLoader, DataLoader_GroupSample, get_dataset_paths, load_action_mapping
from .utils.model import AttentionNetwork
from .utils.analysis import Video, Checkpoint
from .utils.wavenet import WaveNetBackbone
from .home import get_project_base
import argparse
import os
import json
from torch import optim
import torch
from .configs.utils import setup_cfg


def create_videos(vnames: list, label_dict: dict, attrs_saves: list) -> list:
    videos = []
    for i in range(len(vnames)):
        video = Video(vnames[i])
        for k, v in attrs_saves[i].items():
            setattr(video, k, v) # [0] remove batch dim

        for k, l in label_dict.items():
            setattr(video, k, l[i].detach().cpu().numpy())

        videos.append(video)
    return videos

def evaluate(global_step, ckptdir, savedir, net, testloader, pos_weight):
    print("TESTING" + "~"*10)

    ckpt = Checkpoint(global_step+1)

    net.eval()
    with torch.no_grad():
        for batch_idx, (vnames, batch_seq, labels) in enumerate(testloader):
            batch_seq = [ s.cuda() for s in batch_seq ]
            action_labels = labels['action_label'].cuda()

            loss, video_saves = net.forward_and_loss(batch_seq, action_labels, 
                                    transcript=labels['transcript'],
                                    pos_action_weight=pos_weight)

            videos = create_videos(vnames, labels, video_saves)  
            ckpt.add_videos(videos)

    net.train()

    ckpt.average_losses()
    ckpt.compute_metrics()

    string = "loss:%.3f" % ckpt.loss
    print(string)
    string = ""
    for k, v in ckpt.metrics.items():
        string += "%s:%.3f, " % (k, v)
    print(string)


    print('save snapshot ' + str(global_step+1))
    rslt_file = os.path.join(savedir, "%d.gz" % (global_step+1))
    ckpt.save(rslt_file)
    network_file = os.path.join(ckptdir, str(global_step+1) + '.net')
    net.save_model(network_file)
    print()

    return ckpt


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
DATA_LOADED = False


def load_dataset(cfg):
    global DATA_LOADED
    if DATA_LOADED:
        return
    
    DATA_LOADED = True
    global datainfo
    global dataset, trainloader, test_dataset, testloader


    datainfo = get_dataset_paths(cfg.dataset)
    label2index, index2label = load_action_mapping(datainfo.map_fname)
    recipe2index, index2recipe = load_action_mapping(datainfo.recipe_map_fname)
    print("load_data_from", datainfo.dataset_dir)

    with open(datainfo.train_split_fname, 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    with open(datainfo.test_split_fname, 'r') as f:
        test_video_list = f.read().split('\n')[0:-1]

    ### read training data #########################################################
    print('read data...')
    dataset = Dataset( cfg.dataset.name, datainfo.dataset_dir, datainfo.feature_dir, video_list, 
                        label2index, recipe2index, 
                        transpose=datainfo.transpose, feature_type=datainfo.feature_type)
    trainloader = DataLoader_GroupSample(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    ### read testing data #########################################################
    test_dataset = Dataset( cfg.dataset.name, datainfo.dataset_dir, datainfo.feature_dir, test_video_list, 
                        label2index, recipe2index, 
                        transpose=datainfo.transpose, feature_type=datainfo.feature_type )
    testloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)


def already_finished(logdir):
    if os.path.exists(logdir) and os.path.exists(os.path.join(logdir, "FINISH_PROOF")):
        return True
    else:
        return False


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                        help="optional config file", default=[])
parser.add_argument("--set", dest="set_cfgs",
        help="set config keys", default=None, nargs=argparse.REMAINDER,)
parser.add_argument("--runs", nargs="*", default=[-1])

args = parser.parse_args()
args.runs = [ int(r) for r in args.runs ]
BASE = get_project_base()
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    for runid in args.runs:
        cfg = setup_cfg(args.cfg_file, args.set_cfgs, runid)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

        logdir = os.path.join(BASE, "log", cfg.logdir)
        ckptdir = os.path.join(logdir, 'ckpts')
        savedir = os.path.join(logdir, 'save_rslts')

        if already_finished(logdir):
            print('----------------------------------')
            print(cfg.expgroup, "already finished, skip")
            print('----------------------------------')
            continue


        os.makedirs(logdir, exist_ok=True)
        os.makedirs(ckptdir, exist_ok=True)
        os.makedirs(savedir, exist_ok=True)
        print('Saving log at', logdir)

        argSaveFile = os.path.join(logdir, 'args.json')
        with open(argSaveFile, 'w') as f:
            json.dump(cfg, f, indent=True)


        ### load dataset #########################################################
        load_dataset(cfg)
        print('Train dataset', dataset)
        print('Test dataset ', test_dataset)

        ### create action weight according to action frequency in action set ####################
        action_occur_frequency = np.loadtxt(os.path.join(datainfo.dataset_dir, "action_occurence.txt"))
        missing_frequency = 1 - action_occur_frequency
        pos_weight = missing_frequency / action_occur_frequency
        pos_weight[pos_weight==0] = 1 # set weight of Background to normal level      
        pos_weight = torch.FloatTensor(pos_weight).cuda()


        ### create network #########################################################
        backbone = WaveNetBackbone.create(cfg.wavenet)
        net = AttentionNetwork(cfg, backbone, dataset.n_classes)
        if cfg.resume:
            ckpt = torch.load(cfg.resume, map_location="cpu")
            net.load_state_dict(ckpt)
        net.cuda()
        print(net)

        optimizer = optim.SGD(net.parameters(), 
                            lr=cfg.train.lr, 
                            momentum=cfg.train.momentum,
                            weight_decay=cfg.train.weight_decay)

        ### start training #########################################################

        global_step = 0
        start_epoch = 0
        ckpt = Checkpoint(-1)
        for eidx in range(start_epoch, cfg.train.epoch):
    
            for batch_idx, (vnames, batch_seq, labels) in enumerate(trainloader):

                batch_seq = [ s.cuda() for s in batch_seq ]
                action_labels = labels['action_label'].cuda()


                optimizer.zero_grad()
                loss, video_saves = net.forward_and_loss(batch_seq, action_labels, 
                            transcript=labels['transcript'],
                            pos_action_weight=pos_weight)
                net.loss.backward()

                if cfg.train.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.clip_grad_norm)
                optimizer.step()

                videos = create_videos(vnames, labels, video_saves)
                ckpt.add_videos(videos)

                # print some progress information
                if (global_step+1) % cfg.print_every == 0:

                    ckpt.compute_metrics()
                    ckpt.average_losses()

                    string = "Iter%d, " % (global_step+1)
                    _L = len(string)
                    string = "loss:%.3f" % ckpt.loss
                    print(string)

                    string = " " * _L
                    for k, v in ckpt.metrics.items():
                        string += "%s:%.3f, " % (k, v)
                    print(string)

                    ckpt = Checkpoint(-1)

                # test and save model every x iterations
                _save_flag = False
                if global_step == 0 or (global_step+1) % cfg.save_every == 0: 
                    _save_flag = True
                if global_step < cfg.save_every and (global_step+1) % (cfg.save_every // 10) == 0:
                    _save_flag = True # save more frequently at early stage

                if _save_flag:
                    test_ckpt = evaluate(global_step, ckptdir, savedir, net, testloader, pos_weight)


                global_step += 1

        finish_proof_fname = os.path.join(logdir, "FINISH_PROOF")
        open(finish_proof_fname, "w").close()

        print("Finish Stamp")


