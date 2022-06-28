import numpy as np
import pickle
import gzip
from sklearn.metrics import average_precision_score

class Segment():
    def __init__(self, action, start, end):
        assert start >= 0
        self.action = action
        self.start = start
        self.end = end
        self.len = end - start + 1
    
    def __repr__(self):
        return "<%r %d-%d>" % (self.action, self.start, self.end)
    
    def intersect(self, s2):
        s = max([self.start, s2.start])
        e = min([self.end, s2.end])
        return max(0, e-s+1)

    def union(self, s2):
        s = min([self.start, s2.start])
        e = max([self.end, s2.end])
        return e-s+1
    
def parse_label(gt_list):
    current = gt_list[0]
    start = 0
    anno = []
    for i, l in enumerate(gt_list):
        if l == current:
            pass
        else:
            anno.append(Segment(current, start, i-1))
            current = l
            start = i
    anno.append(Segment(current,start, len(gt_list)-1))
    
    return anno

def expand_pred_to_gt_len(pred: np.ndarray, target_len: int) -> np.ndarray:
    import torch

    prediction_tensor = torch.from_numpy(pred).float()
    prediction_tensor = prediction_tensor.view([1, 1, -1])
    resized = torch.nn.functional.interpolate(
        prediction_tensor, size=target_len, mode="nearest"
    ).view(-1)
    resized = resized.detach().numpy()

    return resized

def att_pred_from_onehot_set(attention, aset_onehot):
    """
    generate framewise prediction by contraining to an action set
    the action set is denoted by the onehot vector
    """

    actions = np.where(aset_onehot)[0]
    att = np.zeros_like(attention)
    att[:, actions] = attention[:, actions]
    attention_pred = att.argmax(1)
    return attention_pred

def midpoint_hit(gt_seg, pred_seg):
    unused = np.ones(len(gt_seg), np.bool)
    n_true = len(gt_seg)
    pointer = 0

    TP, FP, FN = 0, 0, 0
    # Go through each segment and check if it's correct.
    for i, pseg in enumerate(pred_seg):
        midpoint = int( (pseg.start + pseg.end)/2 )
        # Check each corresponding true segment
        for j in range(pointer, n_true):
            gseg = gt_seg[j]
            # If the midpoint is in this true segment
            if gseg.start <= midpoint <= gseg.end:
                pointer = j
                # If yes and it's correct
                if (gseg.action == pseg.action):
                    # Only a TP if it's the first occurance. Otherwise FP
                    if unused[j]:
                        TP += 1
                        unused[j] = False
                    else:
                        FP += 1
                # FN if it's wrong class
                else:
                    FN += 1
            elif midpoint < gseg.end:
                break

    midh = float(TP) / (TP+FN+1e-4) 
    
    return midh

def compute_IoU(gt_label, pred, bg_ids=[0]):
    assert isinstance(gt_label, np.ndarray)
    assert isinstance(pred,     np.ndarray)

    unique = np.unique(gt_label).tolist()

    iou = []
    iou_noBG = []
    for i in unique:
        recog_mask = pred == i
        gt_mask = gt_label == i
        union = np.logical_or(recog_mask, gt_mask).sum()
        intersect = np.logical_and(recog_mask, gt_mask).sum() # num of correct prediction
        
        action_iou = intersect / (union+ 1e-8) 

        iou.append(action_iou)
        if i not in bg_ids:
            iou_noBG.append(action_iou)

    iou = np.mean(iou)
    iou_noBG = np.mean(iou_noBG)

    return iou, iou_noBG

class Video():

    def __init__(self, vname=''):
        self.vname = vname
    
    def __str__(self):
        return "< Video %s >" % self.vname

    def __repr__(self):
        return "< Video %s >" % self.vname

class Checkpoint():
    """
    Checkpoint object to help computing metrics and save/load results
    """

    NO_BG_POSTFIX = "_noBG"

    def __init__(self, iteration):

        self.iteration = iteration
        self.losses = None
        self.metrics = None
        self.videos = {}

    def add_videos(self, videos):
        for v in videos:
            self.videos[v.vname] = v

    def drop_videos(self):
        self.videos = {}

    @staticmethod
    def load(fname):
        with gzip.open(fname, 'rb') as fp:
            ckpt = pickle.load(fp)
        return ckpt
    
    def save(self, fname):
        with gzip.open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    def __str__(self):
        return "< Checkpoint[%d] %d videos >" % (self.iteration, len(self.videos))

    def __repr__(self):
        return str(self)

    def _random_video(self):
        vnames = list(self.videos.keys())
        vname = np.random.choice(vnames, 1).item()
        return vname, self.videos[vname]

    def average_losses(self):
        self.loss = np.mean([v.loss for v in self.videos.values()])  
    
    def single_video_set_metrics(self, v, exclude_bg=True):
        postfix = ""
        if exclude_bg:
            action_label = v.action_label[1:] # background id = 0
            action_pred = v.action_pred[1:]
            action_logit = v.action_logit[1:]
            postfix = self.NO_BG_POSTFIX

        if not hasattr(v, 'metrics'):
            v.metrics = {}

        v.metrics.update({
            "ap"+postfix: average_precision_score(action_label, action_logit),
            "acc"+postfix: (action_label == action_pred).mean(),
        })


        return action_label, action_pred

    def single_video_loc_metrics(self, v):

        if not hasattr(v, 'metrics'):
            v.metrics = {}

        noBG = self.NO_BG_POSTFIX

        v.gt_segs = parse_label(v.gt_label)

        # Action Segmentation Task
        seg_pred_short = att_pred_from_onehot_set(v.attention_logit, v.action_pred)
        seg_pred = expand_pred_to_gt_len(seg_pred_short, len(v.gt_label))
        v.seg_pred = seg_pred
        v.seg_segs = parse_label(v.seg_pred)

        iou, iou_noBG = compute_IoU(v.gt_label, seg_pred)
        v.metrics["IoU_seg"] = iou
        v.metrics["IoU_seg" + noBG ] = iou_noBG

        midh = midpoint_hit(v.gt_segs, v.seg_segs)
        v.metrics['MidH_seg'] = midh

        #Action Alignment Task 
        ali_pred_short = att_pred_from_onehot_set(v.attention_logit, v.action_label)
        ali_pred = expand_pred_to_gt_len(ali_pred_short, len(v.gt_label))
        v.ali_pred = ali_pred
        v.ali_segs = parse_label(v.ali_pred)

        iou, iou_noBG = compute_IoU(v.gt_label, ali_pred)
        v.metrics["IoU_ali"] = iou
        v.metrics["IoU_ali" + noBG ] = iou_noBG

        midh = midpoint_hit(v.gt_segs, v.ali_segs)
        v.metrics['MidH_ali'] = midh

        return seg_pred, ali_pred

    def _mof(self, video_list):
        postfix = self.NO_BG_POSTFIX
        gt, seg, ali = [], [], []
        for video in video_list:
            gt.append(video.gt_label)
            seg.append(video.seg_pred)
            ali.append(video.ali_pred)

        gt = np.concatenate(gt)
        seg = np.concatenate(seg)
        ali = np.concatenate(ali)

        fg = gt != 0
        seg_correct = (seg == gt)
        ali_correct = (ali == gt)

        mdict = {
            'Mof_seg' : seg_correct.mean(),
            'Mof_seg'+ postfix : seg_correct[fg].mean(),
            'Mof_ali' : ali_correct.mean(),
            'Mof_ali'+ postfix : ali_correct[fg].mean(),

        }

        return mdict

    def compute_metrics(self):

        for vname, video in self.videos.items():
            video.metrics = {}
            self.single_video_set_metrics(video, exclude_bg=True)
            self.single_video_loc_metrics(video)

        metric_keys = video.metrics.keys()
        self.metrics = { k: np.mean([ v.metrics[k] for v in self.videos.values() ])  
                            for k in metric_keys }

        mof = self._mof(list(self.videos.values()))
        self.metrics.update(mof)

        return self.metrics
