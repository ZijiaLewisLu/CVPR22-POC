from yacs.config import CfgNode as CN

_C = CN()
_C.gpu = 1
_C.mark = "" # for adding addtional note
_C.resume = "" # checkpoint file path
_C.skip_finished = True 
_C.save_every = 10000
_C.print_every = 2000


_C.dataset = CN()
_C.dataset.name = "breakfast"
_C.dataset.split = 1

_C.train = CN()
_C.train.batch_size = 1
_C.train.optimizer = "SGD"
_C.train.epoch = 250
_C.train.lr = 0.1
_C.train.momentum = 0.009
_C.train.weight_decay = 0.000
_C.train.clip_grad_norm = 10.0

_C.wavenet = CN()
_C.wavenet.hidden_size = 32
_C.wavenet.input_size = 64
_C.wavenet.dropout_on_x = 0.05
_C.wavenet.gn_num_groups = 16
_C.wavenet.pooling_levels = [1, 2, 4, 8, 10]
_C.wavenet.output_level = 5

_C.att = CN()
_C.att.hidden_size = 128

_C.thres = CN()
_C.thres.w = 0.0
_C.thres.num_layer = 3
_C.thres.hidden_size = 32

#---------------------------------------------
LossCfg = CN()
LossCfg.w = 0.0

_C.poc_loss = LossCfg.clone()
_C.poc_loss.use_gt = False

_C.att_len_loss = LossCfg.clone()
_C.att_len_loss.min = '' # '' means disable
_C.att_len_loss.max = '' # T/2

_C.att_smooth_loss = LossCfg.clone()

_C.centering_loss = LossCfg.clone()

_C.att_ranking_loss = LossCfg.clone()
_C.att_ranking_loss.nstable = False

def get_cfg_defaults():
    return _C.clone()



