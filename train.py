from torch.utils.data import DataLoader
import torchvision.models as models
from tvqa_dataset import TVQADataset, get_train_transforms, get_test_transforms, TwoAugUnsupervisedTVQADataset, TwoWeakOrAugTVQADataset, OnlyUnknownsTVQADataset
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime
from facenet_pytorch import InceptionResnetV1
from config import default_argument_parser, get_cfg, set_global_cfg, global_cfg
from fvcore.common.file_io import PathManager
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class L2Norm(torch.nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class VGGSupervised(torch.nn.Module):

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=num_classes)

        for param in self.resnet.parameters():
            param.requires_grad = False  # fix the encoder part

        self.resnet.logits = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x


class VGGFaceSupervised(torch.nn.Module):

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=num_classes)

        self.resnet.last_bn = torch.nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False  # fix the encoder part

        self.resnet.logits = torch.nn.Sequential(torch.nn.Linear(in_features=512,
                                                                 out_features=cfg.MODEL.out_features_1, bias=True),
                                                 torch.nn.ReLU(inplace=False),
                                                 torch.nn.Linear(in_features=cfg.MODEL.out_features_1,
                                                                 out_features=cfg.MODEL.out_features_2, bias=True),
                                                 torch.nn.BatchNorm1d(cfg.MODEL.out_features_2,
                                                                      eps=0.001, momentum=0.1, affine=True,
                                                                      track_running_stats=True),
                                                 torch.nn.Linear(in_features=cfg.MODEL.out_features_2,
                                                                 out_features=cfg.MODEL.out_features_3, bias=True),
                                                 )

    def forward(self, x):
        x = self.resnet(x)
        return x


class VGGFacePlus(torch.nn.Module):

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=num_classes)

        self.resnet.last_bn = torch.nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False  # fix the encoder part

        self.resnet.logits = torch.nn.Sequential(torch.nn.Linear(in_features=512,
                                                                 out_features=cfg.MODEL.out_features_1, bias=True),
                                                 torch.nn.ReLU(inplace=False),
                                                 torch.nn.Linear(in_features=cfg.MODEL.out_features_1,
                                                                 out_features=cfg.MODEL.out_features_2, bias=True),
                                                 # torch.nn.ReLU(inplace=False),
                                                 torch.nn.BatchNorm1d(cfg.MODEL.out_features_2,
                                                                      eps=0.001, momentum=0.1, affine=True,
                                                                      track_running_stats=True),
                                                 torch.nn.Linear(in_features=cfg.MODEL.out_features_2,
                                                                 out_features=cfg.MODEL.out_features_3, bias=True),
                                                 # torch.nn.ReLU(inplace=False),
                                                 )
        self.l2norm = L2Norm()

    def forward(self, x):
        x = self.resnet(x)
        return self.l2norm(x)


class VGGFaceSubtitle(torch.nn.Module):

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

        self.resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=num_classes)

        self.resnet.last_bn = torch.nn.Identity()
        self.resnet.logits = torch.nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False  # fix the encoder part

        for param in self.text_model.parameters():
            param.requires_grad = False  # fix the encoder part

        self.face_layer = torch.nn.Linear(in_features=512, out_features=cfg.SSL.face_layer_out_features, bias=True)
        self.sub_layer = torch.nn.Linear(in_features=768, out_features=cfg.SSL.sub_layer_out_features, bias=True)
        self.mix_layer = torch.nn.Linear(in_features=cfg.SSL.mix_layer_in_features,
                                         out_features=cfg.SSL.mix_layer_out_features, bias=True)

        self.l2norm = L2Norm()

    def forward(self, face1, face2, sub1, sub2):
        if face2 is None and sub2 is None:
            x1 = self.resnet(face1)
            sub1 = torch.from_numpy(self.text_model.encode(sub1)).to(self.device)
            if len(sub1.shape) == 1:
                sub1 = sub1[None, :]
            mix = torch.cat([x1, sub1], dim=1)
            if global_cfg.SSL.mix_layer:
                mix = self.mix_layer(mix)
            return self.l2norm(mix)
        else:
            x = torch.cat([face1, face2])
            x = self.resnet(x)

            sub1 = torch.from_numpy(self.text_model.encode(sub1)).to(self.device) # sub1:torch.Size([256, 768])
            sub2 = torch.from_numpy(self.text_model.encode(sub2)).to(self.device)

            #1 layer for image and one for text
            if global_cfg.SSL.face_layer:
                x = self.face_layer(x)
            if global_cfg.SSL.sub_layer:
                sub1 = self.sub_layer(sub1)
                sub2 = self.sub_layer(sub2)

            x1, x2 = x.chunk(2)  # x1: torch.Size([256, 512])
            # concat(x1, sub1) concat(x2, sub2)
            y1 = torch.cat([x1, sub1], dim=1)
            y2 = torch.cat([x2, sub2], dim=1)
            # opposite of chunk
            mix = torch.cat([y1, y2]) # mix: torch.Size([256, 1280])
            # 1/2 layers for both to mix
            if global_cfg.SSL.mix_layer:
                mix = self.mix_layer(mix)

            return self.l2norm(mix).chunk(2)


def train(cfg):
    print("start running: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"cuda is available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(f"{cfg.TRAINING.project_dir}")
    tvqa_train = TVQADataset(series=cfg.TRAINING.series, split="train", transform=get_test_transforms(series=cfg.TRAINING.series))
    # tvqa_train = TwoAugUnsupervisedTVQADataset(split="train", transform=get_train_transforms())
    train_loader = DataLoader(tvqa_train, batch_size=cfg.TRAINING.batch_size, shuffle=True, num_workers=0) #, collate_fn=lambda x: x)

    if cfg.TRAINING.pretrained is True:
        model = VGGFacePlus(cfg, len(tvqa_train.lbl_to_id))
    if cfg.SSL.joint is True:
        model = VGGFaceSubtitle(cfg, len(tvqa_train.lbl_to_id))
    if cfg.SSL.supervised is True:
        model = VGGFaceSupervised(cfg, len(tvqa_train.lbl_to_id))
    if cfg.TRAINING.supervised is True:
        model = VGGSupervised(cfg, len(tvqa_train.lbl_to_id))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.TRAINING.lr)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=cfg.TRAINING.lr,
    #                             momentum=cfg.TRAINING.momentum,
    #                             weight_decay=cfg.TRAINING.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     gamma=cfg.TRAINING.lr_decay_rate,
                                                     milestones=[cfg.TRAINING.lr_decay_epoch])
    criterion = CrossEntropyLoss()
    for epoch in tqdm(range(cfg.TRAINING.epochs)):
        for iteration, data in enumerate(train_loader):
            predictions = model(data['image'].to(device))
            data_mode = cfg.TRAINING.data_mode
            # one_hot = F.one_hot(data["cleansed"].to(device), num_classes=len(tvqa_train.lbl_to_id))
            # loss = criterion(F.softmax(predictions), one_hot)
            loss = criterion(predictions, data[data_mode].to(device))
            # loss = criterion(predictions, data['correct_target_id'].to(device))
            writer.add_scalar("train_loss", loss, epoch*len(train_loader)+iteration)
            for param_group in optimizer.param_groups:
                writer.add_scalar("lr", param_group['lr'], epoch * len(train_loader) + iteration)
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss,
        }, f"{cfg.TRAINING.project_dir}model/epoch_{epoch}.tar")
        print("time: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        if epoch > 0:
            os.remove(f"{cfg.TRAINING.project_dir}model/epoch_{epoch-1}.tar")
    print("end running: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


if __name__ == "__main__":
    # config priority:
    # 1. arguments
    # 2. config.yaml file
    # 3. defaults.py file
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Command line arguments: " + str(args))
    print("Running with full config:\n{}".format(cfg))
    Path(cfg.TRAINING.project_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(cfg.TRAINING.project_dir, "model")).mkdir(parents=True, exist_ok=True)
    config_path = os.path.join(cfg.TRAINING.project_dir, "config.yaml")
    with PathManager.open(config_path, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(os.path.abspath(config_path)))
    set_global_cfg(cfg)
    train(cfg)
