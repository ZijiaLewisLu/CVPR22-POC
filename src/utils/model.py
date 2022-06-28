import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .thresholder import create_thresholder

def torch_order_score(atti, attj):
    assert len(atti.shape) == 1
    assert len(attj.shape) == 1
    cj = torch.cumsum(attj, 0)
    return (atti * (1-cj)).sum()
   
class AttentionNetwork(nn.Module):

    def __init__(self, cfg, backbone, n_classes):
        super(AttentionNetwork, self).__init__()

        self.cfg = cfg 
        self.backbone = backbone
        self.n_classes = n_classes

        self.attention_layer = nn.Linear(cfg.wavenet.hidden_size, n_classes)

        self.action_matrix = nn.Parameter(torch.randn(cfg.wavenet.hidden_size, n_classes))      
        self.action_bias   = nn.Parameter(torch.randn(1, n_classes))
        self.__init_weight__(self.action_matrix, self.action_bias)

        self.thresholder = create_thresholder(self.cfg.thres, cfg.wavenet.hidden_size, n_classes)

    def __init_weight__(self, matrix, bias):
        torch.nn.init.xavier_normal_(matrix)
        torch.nn.init.xavier_normal_(bias)  

    def forward(self, seq):
        """
        seq: 1 x H x T
        """
        self.z = z = self.backbone(seq) # 1, T, H

        self.attention_logit = attention_logit = self.attention_layer(z) # 1, T, C

        self._attention_frame_softmax = F.softmax(attention_logit, dim=2)
        T = attention_logit.shape[1]
        self.attention = self._attention_frame_softmax / T

        # compute video-level action feature
        self.rep_feature = (z[:, :, :, None] * self.attention[:, :, None, :]).sum(1) # 1, H, C

        # predicti action set
        self.action_logit = torch.einsum("bhc,hc->bc", self.rep_feature, self.action_matrix) + self.action_bias 

        feature = self.rep_feature.permute(0, 2, 1) # B, C, H
        self.thres = self.thresholder(feature) # B, C

        return self.attention, self.rep_feature, self.action_logit, self.thres

    def forward_and_loss(self, seq_list, action_label, transcript=None, pos_action_weight=None):

        self.attention_list = attentions = []
        ind_losses  = 0
        video_saves = []
        for i, seq in enumerate(seq_list):
            label = action_label[i].unsqueeze(0)

            self.forward(seq)

            loss = self.single_video_loss(label, pos_action_weight=pos_action_weight)
            attentions.append(self.attention)
            ind_losses += loss

            # if save_attrs:
            save = { 
                "loss" : ind_losses.item(),
                "action_pred" : self.action_pred[0].detach().cpu().numpy(),
                "action_logit": self.action_logit[0].detach().cpu().numpy(),
                'attention_logit': self.attention_logit[0].detach().cpu().numpy(),
            }
            video_saves.append(save)
        
        self.loss = ind_losses / len(seq_list)

        # poc loss
        if self.cfg.poc_loss.w > 0:
            if not self.cfg.poc_loss.use_gt:
                loss = self.poc_loss(attentions, action_label)
            else:
                loss = self.supervised_poc_loss(attentions, action_label, transcript)
            self.loss += self.cfg.poc_loss.w * loss

            for save in video_saves:
                save['loss'] += loss.item()
        
        return self.loss, video_saves

    def single_video_loss(self, action_label, pos_action_weight=None):

        def weight_loss(loss, cfg):
            loss = loss * cfg.w
            return loss

        ############## Video-level Loss
        total_loss = self.ranking_loss(action_label, pos_action_weight)
        thres_loss = self.threshold_loss(action_label, pos_action_weight)
        total_loss += self.cfg.thres.w * thres_loss
        self.action_pred = ( (self.action_logit-self.thres)>0 ).float()

        ############## Frame-level Loss
        if self.cfg.att_smooth_loss.w > 0:
            loss = self.attention_smooth_loss(action_label) 
            total_loss += weight_loss(loss, self.cfg.att_smooth_loss)
        if self.cfg.att_len_loss.w > 0:
            loss = self.attention_length_loss(action_label, self.cfg.att_len_loss)
            total_loss += weight_loss(loss, self.cfg.att_len_loss)
        if self.cfg.att_ranking_loss.w > 0:
            loss = self.attention_ranking_loss(action_label, self.cfg.att_ranking_loss)
            total_loss += weight_loss(loss, self.cfg.att_ranking_loss)
        if self.cfg.centering_loss.w > 0:
            loss = self.centering_loss(action_label)
            total_loss += weight_loss(loss, self.cfg.centering_loss)

        return total_loss

    def save_model(self, network_file):
        torch.save(self.state_dict(), network_file)


    ########################
    ## Video-level Loss
    def ranking_loss(self, action_label, action_weight):
        """
        action_logit 1 x C
        """
        assert action_label.shape[0] == 1
        pos_index = torch.nonzero(action_label[0], as_tuple=True)[0]
        neg_index = torch.nonzero(1-action_label[0], as_tuple=True)[0]

        pos_logit = self.action_logit[:, pos_index] # 1, A
        neg_logit = self.action_logit[:, neg_index] # 1, C-A

        difference = - (torch.t(pos_logit) - neg_logit) # A, C-A 

        difference = torch.clamp(difference, max=50)  # ensure numeric stable 
        difference = torch.exp(difference).sum(dim=1) # A

        weight = action_weight[pos_index]
        weight = weight / weight.mean() # in LSEP, set weight mean to 1
        sumexp = (weight * difference).sum()

        self.action_loss = torch.log(1+sumexp)

        return self.action_loss

    def threshold_loss(self, action_label, action_weight):
        logit = self.action_logit - self.thres
        self.thres_loss = nn.functional.binary_cross_entropy_with_logits(
                                        logit, action_label, weight=action_weight) # shape 1, C
        
        return self.thres_loss
    
    ########################
    ## Frame-level Loss
    def attention_ranking_loss(self, action_label, cfg):
        att = self.attention_logit

        pos_index = torch.nonzero(action_label[0], as_tuple=True)[0]
        neg_index = torch.nonzero(1-action_label[0], as_tuple=True)[0]

        pos_logit = att[:, :, pos_index] # B, T, A
        neg_logit = att[:, :, neg_index] # B, T, C-A

        difference = - (pos_logit.unsqueeze(3) - neg_logit.unsqueeze(2)) # B, T, A, C-A 
        difference = difference.mean(dim=-1)  # B, T, A

        if cfg.nstable:
            # use logsumexp trick to ensure numeric stable
            B, T, _ = difference.shape
            margin = torch.zeros(B, T, 1) + np.log(1)
            margin = margin.to(difference.device)
            difference = torch.cat([difference, margin], dim=2) # B, T, A+1

            max_, indice = torch.max(difference, 2, keepdim=True)

            diff_minux_max = difference - max_ # B, T, A+1
            sumexps = torch.exp(diff_minux_max).sum(2) # B, T
            logsumexps = torch.log(sumexps) + max_[:, :, 0]
            self.att_ranking_loss = logsumexps.mean()

        else:
            sumexps = torch.exp(difference).sum(2) # B, T
            logsumexps = torch.log(1+sumexps)
            self.att_ranking_loss = logsumexps.mean()


        return self.att_ranking_loss

    def attention_length_loss(self, action_label, cfg):
        atten = self._attention_frame_softmax[0]
        pos_index = torch.nonzero(action_label[0], as_tuple=True)[0]
        num_action = len(pos_index)

        T = atten.shape[0]
        length = atten.sum(0)

        def convert(thres):
            if len(thres) == 0: # disable this threshold
                return None
            elif thres.endswith('a'): 
                scale = float(thres[:-1]) * num_action
                return np.ceil(T / scale)
            elif thres.endswith('f'): 
                scale = float(thres[:-1])
                return np.ceil(T / scale)
            else:
                raise ValueError(thres)
                
        loss = 0
        l = length[0] # only apply to background class
        bg_min = convert(cfg.min)
        bg_max = convert(cfg.max)
        if bg_min is not None and l < bg_min: 
            loss += - torch.log(length[0] / bg_min) # maximize
        elif bg_max is not None and l > bg_max:
            loss += - torch.log(bg_max / length[0]) # minimize
            
        self.att_len_loss = loss
        return self.att_len_loss

    def centering_loss(self, action_label):
        attention = self._attention_frame_softmax
        if attention.shape[1] == 1: # In case some video has T=1 after pooling in wavenet
            self.center_loss = torch.tensor(0, device=action_label.device)
            return self.center_loss

        label = action_label[0].clone()
        label[0] = 0 # ignore background
        pos_index = torch.nonzero(label, as_tuple=True)[0]
        attention = attention[:, :, pos_index] # exclude actions not in action set

        action_average_timestamp, offset = self._centering_timestamp_offset(attention)

        loss = (offset**2) * attention
        loss = loss.mean()

        self.center_loss = loss
        return self.center_loss

    def _centering_timestamp_offset(self, attention):
        T = attention.shape[1]
        timestamp = attention.new(np.arange(T)) / (T-1) # normalized timestamp in range 0 to 1
        timestamp = timestamp[None, :, None] # 1, T, 1

        attention_multi_timestamp = attention * timestamp
        action_average_timestamp = attention_multi_timestamp.sum(1, keepdim=True) / ( attention.sum(1, keepdim=True) + 1e-5 ) # B, 1, A 

        timestamp_offset = timestamp - action_average_timestamp # B, T, A 
        return action_average_timestamp, timestamp_offset

    def attention_smooth_loss(self, action_label):
        scores = self._attention_frame_softmax

        a0 = scores[:, :-1, :] 
        a1 = scores[:, 1:,  :] 
        
        diff = torch.abs(a0 - a1)
        diff = diff.mean(dim=1) # B, C
        self.att_smooth_loss = (action_label * diff).mean() # only apply to actions in action set

        return self.att_smooth_loss

    ###########################################
    # pairwise ordering consistency loss
    def poc_loss(self, attentions, action_labels):
        if self.cfg.dataset.name == 'mc2' and len(attentions) == 1: # some recipes in MC2 may only have one video
            self.order_loss = torch.tensor(0, device=action_labels.device)
            return self.order_loss

        assert len(attentions) > 1, len(attentions)
        V = len(attentions)

        # renormalized attention to temporal sum = 1
        unnormalize_attentions = attentions
        attentions = []
        for i, att in enumerate(unnormalize_attentions):
            att = att / ( att.sum(1, keepdim=True) + 1e-5 ) # B, T, C
            attentions.append(att)

        # compute the order scores of each video
        apair_score = {}
        for vid in range(V):
            action_label = action_labels[vid].detach().cpu().numpy()
            action_label[0] = 0 # ignore background class
            actions = np.where(action_label)[0]
            actions.sort()

            C = len(actions)
            if C == 1: # if just one action, skip
                continue

            att = attentions[vid][0] # T, C
            beta = 1 - torch.cumsum(att, 0) # T, C

            O = torch.matmul( torch.t(att), beta ) # C, C  O[i,j] = order(a_i, a_j)

            for i in range(C):
                for j in range(i+1, C):
                    ai, aj = actions[i], actions[j]
                    assert ai < aj, (ai, aj)

                    # since actions are sorted, ai < aj
                    if (ai, aj) not in apair_score:
                        apair_score[(ai, aj)] = [[], []]
                    apair_score[(ai, aj)][0].append(O[ai, aj]) # i before j
                    apair_score[(ai, aj)][1].append(O[aj, ai]) # j before i

        # aggregate and compute loss
        order_loss = 0
        npairs = 0

        for (ai, aj), (score_ij, score_ji) in apair_score.items():
            assert ai < aj, (ai, aj)
            if len(score_ij) == 1: # if just one instance, skip
                continue

            # compute reference ordering
            ref_ij = sum(score_ij) / len(score_ij) # average
            ref_ji = sum(score_ji) / len(score_ji) # average

            for vid_, (oij_v, oji_v) in enumerate(zip(score_ij, score_ji)):
                error_v = 1 - ref_ij * oij_v - ref_ji * oji_v # maximize order score
                order_loss += error_v
                npairs += 1
        
        # return
        if npairs == 0: # return 0 in case there is no common action pairs btw videos in this batch
            self.order_loss = torch.tensor(0, device=action_labels.device)
        else:
            self.order_loss = order_loss / npairs 

        return self.order_loss
   
    def supervised_poc_loss(self, attentions: list, action_labels: torch.Tensor, transcript: list):

        if self.cfg.dataset.name == 'mc2' and len(attentions) == 1:
            self.order_loss = torch.tensor(0, device=action_labels.device)
            return self.order_loss

        V = len(attentions)

        # renormalized attention to temporal sum == 1
        unnormalize_attentions = attentions
        attentions = []
        for i, att in enumerate(unnormalize_attentions):
            att = att / ( att.sum(1, keepdim=True) + 1e-5 ) # B, T, C
            attentions.append(att)
        
        order_loss = 0
        npairs = 0
        for vid in range(V):
            action_label = action_labels[vid].detach().cpu().numpy()
            action_label[0] = 0 # ignore background
            actions = np.where(action_label)[0]
            actions.sort()

            C = len(actions)
            if C == 1: # if just one action, skip
                continue

            att = attentions[vid][0] # T, C
            trans = transcript[vid].detach().cpu().numpy()

            # compute reference ordering from transcript
            T = len(trans)
            true_order_score = {}
            for i in range(T):
                for j in range(i+1, T):
                    ai, aj = trans[i], trans[j]
                    if (ai == 0 or aj == 0): # ignore background
                        continue
                    if ai == aj:
                        continue
                    if (ai, aj) not in true_order_score:
                        true_order_score[(ai, aj)] = []
                    if (aj, ai) not in true_order_score:
                        true_order_score[(aj, ai)] = []

                    true_order_score[(ai, aj)].append(1) # i before j
                    true_order_score[(aj, ai)].append(0) # i before j
            true_order_score = { k: np.mean(v) for k,v in true_order_score.items() }

            beta = 1 - torch.cumsum(att, 0) # T, C
            O = torch.matmul( torch.t(att), beta ) # C, C  O[i,j] = O(a_i, a_j)

            for i in range(C):
                for j in range(i+1, C):
                    ai, aj = actions[i], actions[j]
                    error = 1 - true_order_score[(ai, aj)] * O[ai, aj] \
                                - true_order_score[(aj, ai)] * O[aj, aj]
                    
                    order_loss += error
                    npairs += 1

        if npairs == 0: # in case there is no common action pairs btw videos in this batch
            self.order_loss = torch.tensor(0, device=action_labels.device)
        else:
            self.order_loss = order_loss / npairs 

        return self.order_loss


