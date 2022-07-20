# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from config import CfgNode as CN

_C = CN()

_C.TRAINING = CN()

_C.TRAINING.project_dir = "./output/default/"
_C.TRAINING.data_path = "/home/mayoughi/tvqa_experiment/dataset/friends_frames/"
_C.TRAINING.epochs = 100
_C.TRAINING.lr_decay_epoch = 50
_C.TRAINING.lr_decay_epochs = [50, 75, 90]
_C.TRAINING.pretrained = True
_C.TRAINING.lr = 0.12
_C.TRAINING.batch_size = 256
_C.TRAINING.lr_decay_rate = 0.1
_C.TRAINING.momentum = 0.9
_C.TRAINING.weight_decay = 0.0001
_C.TRAINING.last_commit = "unknown"
_C.TRAINING.supervised = False
_C.TRAINING.data_mode = "correct_target_id"  # cleansed, correct_target_id, weak_label
_C.TRAINING.series = "friends"  # friends, bbt
_C.TRAINING.clustering = "KMeans"  # AgglomerativeClustering, KMeans, MiniBatchKMeans
_C.TRAINING.kmeans_batch_size = 100  # default
_C.TRAINING.exp_type = "normal"  # normal, oracle
_C.TRAINING.ours_or_baseline = "ours"  # ours, baseline

# self-supervised parameters
_C.SSL = CN()
_C.SSL.align_alpha = 2
_C.SSL.unif_t = 2
_C.SSL.align_w = 1
_C.SSL.unif_w = 1
_C.SSL.random_crop = 100
_C.SSL.include_unknowns = True
_C.SSL.joint = False
_C.SSL.face_layer = False
_C.SSL.sub_layer = False
_C.SSL.mix_layer = True
_C.SSL.face_layer_out_features = 512
_C.SSL.sub_layer_out_features = 768
_C.SSL.mix_layer_out_features = 1280
_C.SSL.mix_layer_in_features = 1280

_C.SSL.supervised = False
_C.SSL.epsilon = 0.1 #with probability epsilon pick from closest cluster


_C.MODEL = CN()
_C.MODEL.out_features_1 = 512
_C.MODEL.out_features_2 = 512
_C.MODEL.out_features_3 = 512

_C.GLOBAL = CN()
