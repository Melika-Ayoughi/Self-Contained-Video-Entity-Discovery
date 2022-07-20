import os
import glob

from torchvision import transforms
from tqdm import tqdm
import json
import pysrt
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import math
from pathlib import Path
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import random
import os
import glob
import math
from config import global_cfg
from fuzzywuzzy import fuzz
import pandas as pd
from PIL import Image


def save_json(data, file_path):
    with open(file_path, "w+") as f:
        json.dump(data, f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_train_transforms(series):
    if series == "friends":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(global_cfg.SSL.random_crop, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4313, 0.2598, 0.2205),
                (0.1468, 0.1143, 0.1127),
            ),
        ])
    elif series == "bbt":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(global_cfg.SSL.random_crop, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.3186, 0.2131, 0.1926),
                (0.1695, 0.1326, 0.1284),
            ),
        ])
    return transform


def get_test_transforms(series):
    if series == "friends":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4313, 0.2598, 0.2205),
                (0.1468, 0.1143, 0.1127),
            ),
        ])
    elif series == "bbt":
        transform = transforms.Compose([
            transforms.Resize(size=(160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.3186, 0.2131, 0.1926),
                (0.1695, 0.1326, 0.1284),
            ),
        ])
    return transform


def find_norm_data(dataset):
    # friends mean: tensor([0.4313, 0.2598, 0.2205]) std: tensor([0.1468, 0.1143, 0.1127])
    # bbt mean: tensor([0.3186, 0.2131, 0.1926]) tensor([0.1695, 0.1326, 0.1284])
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        img = data['image']
        batch_samples = img.size(0)
        img = img.view(batch_samples, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


class TVQADataset(data.Dataset):
    def __init__(self, series, split="train", transform=None):
        self.entity_recognition_model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english")
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") #todo: preload the model
        self.label_list = [
            "O",  # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",  # Beginning of a person's name right after another person's name
            "I-PER",  # Person's name
            "B-ORG",  # Beginning of an organisation right after another organisation
            "I-ORG",  # Organisation
            "B-LOC",  # Beginning of a location right after another location
            "I-LOC"  # Location
        ]
        self.split = split
        self.series = series
        self.transform = transform
        self.project_dir = "./"
        self.all_subtitles_loc = self.project_dir + "dataset/tvqa_subtitles/"

        # self.subtitle_json = self.all_subtitles_loc + f"subtitle_cache_friends_s01e{episode}.json"
        # for i in range(1, 6):
        #     srt_data = self.load_srt(self.all_subtitles_loc, self.all_subtitles_loc + f"subtitle_cache_friends_s01e0{i}.json")
        #     self.dataset_dict = self.prepare_tvqa_json(srt_data, dataset_path=self.project_dir + f"dataset/train_episodes/friends_dict_s01e0{i}_extendedbb.json")
        # self.dataset_dict = self.clean_dict(self.dataset_dict, path=self.project_dir + f"dataset/friends_dict.json")
        if global_cfg.TRAINING.exp_type == "normal":

            if self.series == "bbt":
                self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/{self.series}_{self.split}_annotations.json")
            elif self.series == "friends":
                self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/{self.split}_annotations.json")
            #     if self.split == "train":
            #         self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/new_mix_{self.series}_annotations.json")
            #     elif self.split == "test":
            #         self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/new_tvqa_plus_test_annotations.json")
            # elif self.series == "friends":
            #     self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/{self.split}_annotations.json")
                # self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/weak_all_{self.series}_{self.split}_annotations.json")
        elif global_cfg.TRAINING.exp_type == "oracle":
            if self.series == "bbt":
                if global_cfg.TRAINING.ours_or_baseline == "baseline":
                    if self.split == "train":
                        self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/bbt_1_8.json")
                        # self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/weak_bbt_train_annotations.json")
                    elif self.split == "test":
                        self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/bbt_9_10.json")
                elif global_cfg.TRAINING.ours_or_baseline == "ours":
                    if self.split == "train":
                        self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/{self.series}_{self.split}_annotations.json")
                    elif self.split == "test":
                        self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/bbt_9_10.json")
        self.lbl_to_id = self.load_labels(self.anns,  lbl_path=self.project_dir + f"dataset/{self.series}_lbl_to_id.json")
        self.id_to_lbs = {id: lbl for (lbl, id) in self.lbl_to_id.items()}

    def is_unknown(self, ann):
        if not ann['name']:
            return True
        if len(set(ann['name']).intersection(set(self.lbl_to_id.keys()))) > 0:
            return False
        else:
            return True

    def stitch_tokens_in_ann(self, anns):
        for ann in anns.values():
            if ann["name"]:
                i = 0
                while i < len(ann["name"]):
                    weak_lbl = ann["name"][i]
                    if weak_lbl.startswith("##"):
                        if i != 0:
                            print(weak_lbl, ann["name"][i - 1])
                            ann["name"][i - 1] = ann["name"][i - 1] + weak_lbl[2:]
                            ann["name"].remove(weak_lbl)
                            print(ann["name"][i - 1])
                        else:
                            i += 1
                    else:
                        print(weak_lbl)
                        i += 1
        return anns

    def stitch_tokens_in_dict(self, dataset_dict):
        for data in dataset_dict.values():
            if data["names"]:
                i = 0
                while i < len(data["names"]):
                    weak_lbl = data["names"][i]
                    if weak_lbl.startswith("##"):
                        if i != 0:
                            print(weak_lbl, data["names"][i - 1])
                            data["names"][i - 1] = data["names"][i - 1] + weak_lbl[2:]
                            data["names"].remove(weak_lbl)
                            print(data["names"][i - 1])
                        else:
                            i += 1
                    else:
                        print(weak_lbl)
                        i += 1
        return dataset_dict

    def build_tvqa_data(self):
        # self.lbl_to_id = {}
        # self.build_tvqa_data()
        srt_data = self.load_srt(self.all_subtitles_loc, self.subtitle_json)
        dataset_dict = self.prepare_tvqa_json(srt_data)
        clean_dataset_dict = self.clean_dict(dataset_dict, path=self.project_dir + f"dataset/clean_friends_dict.json")
        self.anns = self.load_anns(clean_dataset_dict, ann_path=self.project_dir + f"dataset/clean_annotations.json")
        self.lbl_to_id = self.load_labels(self.anns, lbl_path=self.project_dir + "dataset/lbl_to_id.json")
        self.split_train_val_test(self.anns, path=self.project_dir + "dataset/")
        self.create_hist(file_name="all_annotations_hist")

    def clean_anns(self, anns, path, lbl_type):
        if os.path.exists(path):
            print("Found clean dictionary cached, loading ...")
            return load_json(path)
        all_labels = []
        lbl_to_freq = {}
        to_be_deleted_lbls = set([])
        for ann in tqdm(anns.values()):
            all_labels.append(ann[lbl_type])

        print(all_labels)
        for lbl in set(all_labels):
            lbl_to_freq[lbl] = 0

        # calculate label frequencies:
        for lbl in all_labels:
            lbl_to_freq[lbl] += 1

        for lbl in lbl_to_freq.keys():
            print(f"label: {lbl} frequency:{lbl_to_freq[lbl]}")

        # sort based on frequencies
        sort_dict = dict(sorted(lbl_to_freq.items(), key=lambda item: item[1], reverse=True))
        for i, lbl in enumerate(sort_dict.keys()):
            if lbl_type == 'target_name':
                if i >= 7:  # includes "Unknown"
                    to_be_deleted_lbls.add(lbl)
            elif lbl_type == 'name':
                if i >= 6:  # doesn't include "Unknown"
                    to_be_deleted_lbls.add(lbl)
        print(f"deleted labels: {to_be_deleted_lbls}")

        # delete corresponding annotations
        for ann in tqdm(anns.values()):
            print(f"before: {ann[lbl_type]}")
            # [x for x in array1 if x not in array2]
            # delete less frequents:
            # dataset[img]['names'] = [x for x in dataset[img]['names'] if x not in to_be_deleted_lbls]
            # keeping less frequent as unknowns:
            ann[lbl_type] = ann[lbl_type] if ann[lbl_type] not in to_be_deleted_lbls else 'Unknown'
            # dataset[img]['names'] = list(set(dataset[img]['names']) - to_be_deleted_lbls)
            print(f"after: {ann[lbl_type]}")

        save_json(anns, path)
        return anns

    def clean_dict(self, dataset, path):
        if os.path.exists(path):
            print("Found clean dictionary cached, loading ...")
            return load_json(path)

        all_labels = []
        lbl_to_freq = {}
        to_be_deleted_lbls = set([])

        for img in tqdm(dataset.keys()):
            # if list is not empty
            if dataset[img]['names']:
                all_labels.extend(dataset[img]['names'])  # add a list to list

        print(all_labels)
        for lbl in set(all_labels):
            lbl_to_freq[lbl] = 0

        # calculate label frequencies:
        for lbl in all_labels:
            lbl_to_freq[lbl] += 1

        for lbl in lbl_to_freq.keys():
            print(f"label: {lbl} frequency:{lbl_to_freq[lbl]}")

        for lbl, freq in lbl_to_freq.items():
            if freq < 580:
                to_be_deleted_lbls.add(lbl)
        print(f"deleted labels: {to_be_deleted_lbls}")

        # delete corresponding annotations
        for img in tqdm(dataset.keys()):
            print(f"before: {dataset[img]['names']}")
            # [x for x in array1 if x not in array2]
            # delete less frequents:
            # dataset[img]['names'] = [x for x in dataset[img]['names'] if x not in to_be_deleted_lbls]
            # keeping less frequent as unknowns:
            dataset[img]['names'] = [x if x not in to_be_deleted_lbls else 'Unknown' for x in dataset[img]['names']]
            # dataset[img]['names'] = list(set(dataset[img]['names']) - to_be_deleted_lbls)
            print(f"after: {dataset[img]['names']}")
        save_json(dataset, path)
        return dataset

    def __getitem__(self, index):
        ann = self.anns[str(index)]
        if global_cfg.TRAINING.exp_type == "oracle":
            dir = f"./dataset/new_tvqa_plus_{self.series}_frames_onlyfaces/"
        elif global_cfg.TRAINING.exp_type == "normal":
            dir = "./dataset/" + ann['series'] + "_frames/"
        image = Image.open(os.path.join(dir, ann['face']))
        if self.transform is not None:
            image = self.transform(image)
        if global_cfg.TRAINING.exp_type == "oracle":
            dict = {
                "image": image,
                "clip": ann["clip"],
                "series": ann["series"],
                "face": ann["face"],
                "correct_target_name": ann["name"],
                "correct_target_id": self.lbl_to_id[ann["name"]],
                "bbox": ann["bbox"],
                # "weak_label": self.lbl_to_id[ann['weak_lbls'][0]],
            }
        elif global_cfg.TRAINING.exp_type == "normal":
            dict = {
                    # "image": image,
                    "clip": ann["clip"],
                    "series": ann["series"],
                    "face": ann["face"],
                    "subtitle": ann["subtitle"],
                    }

            if self.split == "train":
                # if global_cfg.TRAINING.series == "friends":
                target = []
                if ann['name']:
                    for name in ann['name']:
                        if name in self.lbl_to_id.keys():
                            target.append(self.lbl_to_id[name])
                        else:
                            target.append(self.lbl_to_id["Unknown"])
                # else:
                #     target.append(self.lbl_to_id["Unknown"])
                # else:
                #     target = self.lbl_to_id[ann['name']]
                dict["weak_id"] = target
                # dict["weak_id"] = target[0]
                dict["weak_name"] = ann['name']
                dict["cleansed"] = ann["cleansed"]
                if self.series == "bbt":
                    dict["bbox"] = ann["bbox"]
                    dict["face_points"] = ann["face_points"]

            elif self.split == "test":
                if self.series == "friends":
                    dict["correct_target_name"] = ann["target_name"]
                    dict["correct_target_id"] = self.lbl_to_id[ann["target_name"]]
                elif self.series == "bbt":
                    dict["correct_target_name"] = ann["name"]
                    dict["correct_target_id"] = self.lbl_to_id[ann["name"]]

        return dict
        # target = self.lbl_to_id[ann['name']]
        # target = []
        # if ann['name']:
        #     for name in ann['name']:
        #         if name in self.lbl_to_id.keys():
        #             target.append(self.lbl_to_id[name])
        #         else:
        #             target.append(self.lbl_to_id["Unknown"])

        # return {"image": image,
        # "weak_label": target,
        # "correct_target_name": ann["target_name"],
        # "correct_target_id": self.lbl_to_id[ann["target_name"]],
        # "weak_label": [self.lbl_to_id[name] for name in ann['name'] if ann['name']],
        # }
        # "weak_label": [self.lbl_to_id[name] for name in ann['name'] if ann['name']]}
        # , "weak_label": target, "correct_target_id": ann["target_id"]}

    def __len__(self):
        return len(list(self.anns.keys()))

    def load_labels(self, anns, lbl_path, lbl_type="name"):

        if os.path.exists(lbl_path):
            print("Found labels cache, loading ...")
            return load_json(lbl_path)

        labels = set([])
        lbl_to_id = {}
        if self.split == "test" or self.split == "dev":
            if lbl_type == "target_name":
                for ann in anns.values():
                    labels.add(ann[lbl_type])
            elif lbl_type == "name":
                for ann in anns.values():
                    if ann[lbl_type]:
                        for lbl in ann[lbl_type]:
                            labels.add(lbl)
        elif self.split == "train":
            for ann in anns.values():
                labels.add(ann[lbl_type])

        for idx, lbl in enumerate(labels):
            lbl_to_id[lbl] = idx

        save_json(lbl_to_id, lbl_path)
        return lbl_to_id

    def load_anns(self, dataset, ann_path):

        if os.path.exists(ann_path):
            print("Found annotation cache, loading ...")
            return load_json(ann_path)

        anns = {}
        all_faces = []
        annid = 0
        for img in tqdm(dataset.keys()):
            # if there is at least one face and one name (not empty)
            if dataset[img]['faces'] and dataset[img]['names']:
                for face, bbox, face_landmark in zip(dataset[img]['faces'], dataset[img]['bbox'], dataset[img]['face_points']):
                    for name in dataset[img]['names']:
                        anns[annid] = {'face': face,
                                       'name': name,
                                       'img': dataset[img]['img'],
                                       'subtitle': dataset[img]['subtitle'],
                                       'clip': dataset[img]['clip'],
                                       'series': dataset[img]['series'],
                                       'bbox': bbox,
                                       'face_points': face_landmark}
                        annid += 1
        save_json(anns, ann_path)
        return load_json(ann_path)

    def load_anns_test(self, dataset, ann_path):

        if os.path.exists(ann_path):
            print("Found annotation cache, loading ...")
            return load_json(ann_path)

        anns = {}
        annid = 0
        for img in tqdm(dataset.keys()):
            # if there is at least one face and one name (not empty)
            if dataset[img]['faces']:
                for face in dataset[img]['faces']:
                    anns[annid] = {'face': face, 'name': dataset[img]['names'], 'img': dataset[img]['img'],
                                   'subtitle': dataset[img]['subtitle'],
                                   'clip': dataset[img]['clip'],
                                   'series': dataset[img]['series']}
                    annid += 1
        save_json(anns, ann_path)
        return load_json(ann_path)

    def find_names(self, sequence):
        # Bit of a hack to get the tokens with the special tokens
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(sequence)))
        inputs = self.tokenizer.encode(sequence, return_tensors="pt")

        outputs = self.entity_recognition_model(inputs)[0]  # .logits
        predictions = torch.argmax(outputs, dim=2)
        # todo: should also include location names and other things
        # name_to_label = [(token, self.label_list[prediction]) for token, prediction in
        #                  zip(tokens, predictions[0].numpy()) if
        #                  self.label_list[prediction] == "B-PER" or self.label_list[prediction] == "I-PER"]
        name_to_other_labels = []
        name_to_label = []
        for token, prediction in zip(tokens, predictions[0].numpy()):
            if self.label_list[prediction] == "B-PER" or self.label_list[prediction] == "I-PER":
                name_to_label.append((token, self.label_list[prediction]))
            elif self.label_list[prediction] in ["B-MISC", "I-MISC", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]:
                name_to_other_labels.append((token, self.label_list[prediction]))

        return name_to_label, name_to_other_labels

    def load_srt(self, srt_dir, srt_cache_path):
        """
        return: A python dict, the keys are the video names, the entries are lists,
                each contains all the text from a .srt file
        sub_times are the start time of the sentences.
        """
        if os.path.exists(srt_cache_path):
            print("Found srt data cache, loading ...")
            return load_json(srt_cache_path)
        print("Loading srt files from %s ..." % srt_dir)
        # srt_paths = glob.glob(os.path.join(srt_dir, "friends_s01e02_seg02_clip_*.srt"))

        srt_paths = glob.glob(os.path.join(srt_dir, f"friends_s01e{self.episode}_seg0*.srt"))
        # srt_paths = glob.glob(os.path.join(srt_dir, f".srt"))
        name2sub_text = {}
        name2sub_face = {}
        name2sub_time = {}
        name2sub_time_end = {}
        for i in tqdm(range(len(srt_paths))):
            subs = pysrt.open(srt_paths[i], encoding="iso-8859-1")
            if len(subs) == 0:
                subs = pysrt.open(srt_paths[i])

            text_list = []
            name_list = []
            sub_time_list = []
            sub_time_list_end = []
            for j in range(len(subs)):
                cur_sub = subs[j]
                cur_str = cur_sub.text
                cur_str = "(<UNKNAME>:)" + cur_str if cur_str[0] != "(" else cur_str
                cur_str = cur_str.replace("\n", " ")
                names = self.find_names(cur_str)
                text_list.append(cur_str)
                name_list.append(names)
                sub_time_list.append(
                    60 * cur_sub.start.minutes + cur_sub.start.seconds + 0.001 * cur_sub.start.milliseconds)
                sub_time_list_end.append(
                    60 * cur_sub.end.minutes + cur_sub.end.seconds + 0.001 * cur_sub.end.milliseconds)

            key_str = os.path.splitext(os.path.basename(srt_paths[i]))[0]
            name2sub_text[key_str] = text_list
            name2sub_face[key_str] = name_list
            name2sub_time[key_str] = sub_time_list
            name2sub_time_end[key_str] = sub_time_list_end

        srt_data = {"sub_text": name2sub_text,
                    "sub_face": name2sub_face,
                    "sub_time": name2sub_time,
                    "sub_time_end": name2sub_time_end}
        save_json(srt_data, srt_cache_path)
        return load_json(srt_cache_path)  # we do this because the ints will turn to string 0 -> '0'

    def prepare_tvqa_json(self, srt_data, dataset_path):
        if srt_data is None:
            return
        series_list = ["castle", "friends", "grey", "house", "met", "bbt"]
        series_directory = self.project_dir + f"dataset/frames_hq/{series_list[5]}_frames/"
        save_directory = self.project_dir + f"/dataset/{series_list[5]}_frames/"

        Path(save_directory).mkdir(parents=True, exist_ok=True)

        if os.path.exists(dataset_path):
            print("Found dataset cache, loading ...")
            return load_json(dataset_path)

        mtcnn = MTCNN(keep_all=True)
        i = 0
        dataset_dict = {}
        for clip in tqdm(srt_data['sub_text'].keys()):
            if clip.startswith(""):
                for t, names in enumerate(srt_data['sub_face'][clip]):
                    # print(t, text)
                    # print(t, srt_data['sub_time'][clip][t])
                    # matching subtitles to all frames in that time frame
                    frame_num_begin = math.ceil(srt_data['sub_time'][clip][t] * 3)
                    frame_num_end = math.ceil(srt_data['sub_time_end'][clip][t] * 3)
                    frame_num = frame_num_begin
                    # for frame_num in range(frame_num_begin, frame_num_end):
                    # print(t, clip, srt_data['sub_time'][clip][t], frame_num, str(frame_num).zfill(5)+".jpg")
                    img_str = str(frame_num).zfill(5)
                    clip_dir = series_directory + clip
                    if not names:
                        names = []
                    else:
                        names = [name_lbl[0] for name_lbl in names]

                    # this is in case the frame number does not exist(usually happens for the last frames in folder)
                    try:
                        # make a function to draw
                        img = Image.open(os.path.join(clip_dir, img_str + ".jpg"))
                        boxes, probs, points = mtcnn.detect(img, landmarks=True)

                        faces = []
                        if boxes is not None:
                            img_draw = img.copy()
                            draw = ImageDraw.Draw(img_draw)
                            for f, (box, point) in enumerate(zip(boxes, points)):
                                draw.rectangle(box.tolist(), width=5)
                                faces.append("{}_{}_{}.png".format(clip, img_str, f))
                                box[0] = box[0] - 20
                                box[1] = box[1] - 20
                                box[2] = box[2] + 20
                                box[3] = box[3] + 20
                                # for p in point:
                                #     draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
                                extract_face(img, box,
                                             save_path=save_directory + "{}_{}_{}.png".format(clip, img_str, f))
                        if points is not None:
                            points = points.tolist()
                        if boxes is not None:
                            boxes = boxes.tolist()
                        dataset_dict[i] = {"series": "bbt", "clip": clip, "img": img_str, "names": names,
                                           "subtitle": srt_data['sub_text'][clip][t], "faces": faces,
                                           "face_points": points, "bbox": boxes}
                        i += 1
                    except OSError as e:
                        print(e)
                        continue
        dataset_dict = self.stitch_tokens_in_dict(dataset_dict)
        save_json(dataset_dict, dataset_path)
        return load_json(dataset_path)

    def create_hist(self, file_name, lbl_type):
        # lbl_type can be target_name(correct) or name(weak)
        num_classes = len(self.lbl_to_id.keys())
        hist_bins = np.arange(num_classes + 1)
        histogram = np.zeros((num_classes,), dtype=np.int)
        if self.split == "test" or self.split == "dev":
            if lbl_type == "target_name":
                classes = [self.lbl_to_id[ann[lbl_type]] for ann in self.anns.values()]
            elif lbl_type == "name":
                classes = []
                for ann in self.anns.values():
                    if ann[lbl_type]:
                        for lbl in ann[lbl_type]:
                            classes.append(self.lbl_to_id[lbl])
        elif self.split == "train":
            classes = [self.lbl_to_id[ann[lbl_type]] for ann in self.anns.values()]
        histogram += np.histogram(classes, bins=hist_bins)[0]
        ind_sorted = np.argsort(histogram)[::-1]
        bins = range(num_classes)
        fig = plt.figure(figsize=(10, 8))

        plt.bar(bins, height=histogram[ind_sorted], color='#3DA4AB')
        # plt.yscale("log")
        plt.ylabel("#instances", rotation=90)
        id_to_lbl = {id: lbl for (lbl, id) in self.lbl_to_id.items()}
        class_names = [id_to_lbl[ind] for ind in ind_sorted]
        plt.xticks(bins, np.array(class_names), rotation=90, fontsize=10)
        fig.savefig(os.path.join("", f"{file_name}.pdf"))
        plt.clf()

    def create_cooccurance_matrix(self):
        coocur_matrix = np.zeros((len(self.lbl_to_id), len(self.lbl_to_id)), np.float64)
        for ann in self.dataset_dict.values():
            for i, n1 in enumerate(ann["names"]):
                for j, n2 in enumerate(ann["names"]):
                    if i != j:
                        coocur_matrix[self.lbl_to_id[n1]][self.lbl_to_id[n2]] += 1

        id_to_lbl = {id: lbl for (lbl, id) in self.lbl_to_id.items()}
        num_classes = len(self.lbl_to_id.keys())
        class_names = [id_to_lbl[ind] for ind in range(num_classes)]
        plt.imshow(coocur_matrix, cmap='plasma', interpolation='nearest')

        plt.xticks(range(num_classes), np.array(class_names), rotation=90, fontsize=6)
        plt.yticks(range(num_classes), np.array(class_names), rotation=0, fontsize=6)

        # Plot a colorbar with label.
        cb = plt.colorbar()
        cb.set_label("Number of co-occurrences")

        # Add title and labels to plot.
        plt.title("Co-occurrence of named entities in subtitles only")
        plt.xlabel('Named Entities')
        plt.ylabel('Named Entities')
        plt.savefig('cooccurance_matrix_plasma.pdf')
        plt.clf()

    def split_train_val_test(self, anns, path):
        # want to use targets to do stratified split
        y = []
        for ann in anns.values():
            y.append(self.lbl_to_id[ann["name"]])
        X = list(anns.keys())
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=49, shuffle=True,
                                                            stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.88, random_state=50,
                                                          shuffle=True, stratify=y_train)

        # use these three subset indices to save the files for train, val, test
        train_anns = {str(i): anns[idx] for i, idx in enumerate(X_train)}
        val_anns = {str(i): anns[idx] for i, idx in enumerate(X_val)}
        test_anns = {str(i): anns[idx] for i, idx in enumerate(X_test)}
        save_json(train_anns, os.path.join(path, "train_annotations.json"))
        save_json(val_anns, os.path.join(path, "val_annotations.json"))
        save_json(test_anns, os.path.join(path, "test_annotations.json"))

    def split_test_dev_sets(self):
        annotation_paths = glob.glob(os.path.join("./dataset/test_episodes/", f"labeled_annotations_test_s01e*.json"))
        all_test_anns = {}
        i = 0
        for ann_path in annotation_paths:
            per_episode_ann = load_json(ann_path)
            print(len(per_episode_ann.keys()))
            for ann in per_episode_ann.values():
                all_test_anns[i] = ann
                i += 1

        all_inds = list(range(len(all_test_anns)))
        random.shuffle(all_inds)
        test_inds = all_inds[:math.ceil(len(all_inds) / 2)]
        dev_inds = all_inds[math.ceil(len(all_inds) / 2) + 1:]
        test_set = {i: all_test_anns[ind] for i, ind in enumerate(test_inds)}
        dev_set = {i: all_test_anns[ind] for i, ind in enumerate(dev_inds)}
        save_json(test_set, "./dataset/test.json")
        save_json(dev_set, "./dataset/dev.json")

    def split_train_set(self, n=10):
        annotation_paths = glob.glob(os.path.join("./dataset/train_episodes/", f"all_annotations_s01e*.json"))
        all_train_anns = {}
        i = 0
        for ann_path in annotation_paths:
            per_episode_ann = load_json(ann_path)
            print(len(per_episode_ann.keys()))
            for ann in per_episode_ann.values():
                all_train_anns[i] = ann
                i += 1
        save_json(all_train_anns, "./dataset/train.json")

        splitted_faces = {}
        for ep in range(6, 25):
            if ep < 10:
                episode = "0" + str(ep)
            elif ep == 16:
                episode = "16-17"
            elif ep == 17:
                continue
            else:
                episode = str(ep)
            faces = load_json(f"./dataset/train_episodes/faces_s01e{episode}.json")
            random.shuffle(faces)
            for i, face_split in enumerate(np.array_split(np.array(faces), n)):
                if episode == "06":
                    splitted_faces[i] = []
                splitted_faces[i].extend(face_split)

        train_anns = {}
        for i in range(n):
            train_anns[i] = {}
        for i, face_chunk in enumerate(splitted_faces.values()):
            j = 0
            for face in face_chunk:
                for ann in all_train_anns.values():
                    if ann["face"] == face:
                        train_anns[i][j] = ann
                        j += 1
        for i in range(n):
            save_json(train_anns[i], f"./dataset/part{i}_train.json")

    def create_confusion_matrix(self):
        import pandas as pd
        self.lbl_to_id["Unknown"] = len(self.lbl_to_id)
        confusion_matrix = np.zeros((len(self.lbl_to_id), len(self.lbl_to_id)), np.float64)
        exp = "tvqa_exp_overfit_notshuffle_split"
        df = pd.read_excel(f"./output/{exp}/test_data.xls", usecols="A,B,C")  # weak label, correct label, prediction
        for i in range(len(df['correct label']) - 1):  # last row is nothing
            confusion_matrix[self.lbl_to_id[df['correct label'][i]]][self.lbl_to_id[df['prediction'][i]]] += 1
            print(f"label: {df['correct label'][i]}, predicitons: {df['prediction'][i]}")

        id_to_lbl = {id: lbl for (lbl, id) in self.lbl_to_id.items()}
        num_classes = len(self.lbl_to_id.keys())
        class_names = [id_to_lbl[ind] for ind in range(num_classes)]
        plt.imshow(confusion_matrix, cmap='plasma', interpolation='nearest')

        plt.xticks(range(num_classes), np.array(class_names), rotation=90, fontsize=6)
        plt.yticks(range(num_classes), np.array(class_names), rotation=0, fontsize=6)

        # Plot a colorbar with label.
        cb = plt.colorbar()
        cb.set_label("Number of predictions")

        # Add title and labels to plot.
        plt.title("Confusion Matrix for predictions and correct labels")
        plt.xlabel('Correct Label')
        plt.ylabel('Predicted Label')
        plt.savefig('confusion_matrix.pdf')
        plt.clf()

    def label_annotations(self):
        import pandas as pd
        dataset_loc = "./dataset/excel"
        df = pd.read_excel(f"{dataset_loc}/new_all_data.xls", usecols="A,B")
        self.lbl_to_id["Unknown"] = len(self.lbl_to_id)
        for i, ann in enumerate(self.anns.values()):
            print(df['correct label'][i], self.lbl_to_id[df['correct label'][i]])
            ann["target_name"] = df['correct label'][i]
            ann["target_id"] = self.lbl_to_id[df['correct label'][i]]
        save_json(self.anns, file_path=self.project_dir + f"dataset/labeled_clean_annotations.json")

    def label_annotations_for_test(self, episode):
        import pandas as pd
        dataset_loc = "./dataset/excel"
        df = pd.read_excel(f"{dataset_loc}/s01e{episode}_new.xls", usecols="A,B,I")
        # df = pd.read_excel(f"{dataset_loc}/missing_faces.xls", usecols="A,E")
        # self.lbl_to_id["Unknown"] = len(self.lbl_to_id)
        for ann in self.anns.values():
            # if ann["target_name"] == "Dunno":
            filtered_df = df.loc[df['face_loc'] == ann['face']]
            if not filtered_df['correct label'].empty:
                # print(filtered_df['correct label'].iloc[0])
                # might have more matchings but the correct label is the same for all
                ann["target_name"] = filtered_df['correct label'].iloc[0]
                # ann["target_id"] = self.lbl_to_id[df['correct label'][i]]
            else:
                print(f"did not find face {ann['face']} in excel")
                ann["target_name"] = "Dunno"
        save_json(self.anns,
                  file_path=self.project_dir + f"dataset/test_episodes/really_labeled_annotations_test_s01e{episode}.json")


class OnlyUnknownsTVQADataset(TVQADataset):

    def __len__(self):
        return len(self.anns_no_unknowns)

    def __getitem__(self, index):
        ann1 = self.anns_no_unknowns[index]
        dir = "./dataset/" + ann1['series'] + "_frames/"
        print(f"directory:{dir}, cwd: {os.getcwd()}")
        image1 = Image.open(os.path.join(dir, ann1['face']))

        dataset_len = len(self)
        random_index = int(np.random.random() * dataset_len)
        ann2 = self.anns_no_unknowns[random_index]


        image2 = Image.open(os.path.join(dir, ann2['face']))

        img1, img2 = self.transform(image1), self.transform(image2)
        dict1, dict2 = {}, {}

        if self.split == "train":
            # target = []
            # if ann1['name']:
            #     for name in ann1['name']:
            #         if name in self.lbl_to_id.keys():
            #             target.append(self.lbl_to_id[name])
            #         else:
            #             target.append(self.lbl_to_id["Unknown"])
            # target = self.lbl_to_id[ann['name']]
            dict1 = {"image": img1,
                     "subtitle": ann1["subtitle"],
                     }
            dict2 = {"image": img2,
                     "subtitle": ann2["subtitle"],
                     }
        elif self.split == "test":
            dict1 = {"image": img1,
                     "correct_target_name": ann1["target_name"],
                     "correct_target_id": self.lbl_to_id[ann1["target_name"]],
                     "subtitle": ann1["subtitle"],
                     }
            dict2 = {"image": img2,
                     "correct_target_name": ann2["target_name"],
                     "correct_target_id": self.lbl_to_id[ann2["target_name"]],
                     "subtitle": ann2["subtitle"],
                     }

        return dict1, dict2


class TwoWeakOrAugTVQADataset(TVQADataset):
    def get_random_sample(self, include_unknowns=True):
        dataset_len = len(self)
        while True:
            random_index = int(np.random.random() * dataset_len)
            ann = self.anns[str(random_index)]
            if include_unknowns is False:
                if self.is_unknown(ann):
                    continue
            dir = "./dataset/" + ann['series'] + "_frames/"
            image = Image.open(os.path.join(dir, ann['face']))
            break
        return ann, image

    def __getitem__(self, index):
        ann1 = self.anns[str(index)]
        dir = "./dataset/" + ann1['series'] + "_frames/"
        image1 = Image.open(os.path.join(dir, ann1['face']))
        if global_cfg.SSL.include_unknowns is False:
            if self.is_unknown(ann1):
                ann1, image1 = self.get_random_sample(include_unknowns=global_cfg.SSL.include_unknowns)

        # Get a random sample that has the same weak label
        while True:
            #todo: unknowns together
            if self.is_unknown(ann1):  # if no weak label exists, do augmentation
                ann2 = ann1
                image2 = Image.open(os.path.join(dir, ann2['face']))
                break
            ann2, image2 = self.get_random_sample(include_unknowns=global_cfg.SSL.include_unknowns)
            if len(set(ann1['name']).intersection(set(ann2['name']))) > 0:
                break

        img1, img2 = self.transform(image1), self.transform(image2)
        dict1, dict2 = {}, {}

        if self.split == "train":
            # target = []
            # if ann1['name']:
            #     for name in ann1['name']:
            #         if name in self.lbl_to_id.keys():
            #             target.append(self.lbl_to_id[name])
            #         else:
            #             target.append(self.lbl_to_id["Unknown"])
            # target = self.lbl_to_id[ann['name']]
            dict1 = {"image": img1,
                     "subtitle": ann1["subtitle"],
                     }
            dict2 = {"image": img2,
                     "subtitle": ann2["subtitle"],
                     }
        elif self.split == "test":
            dict1 = {"image": img1,
                     "correct_target_name": ann1["target_name"],
                     "correct_target_id": self.lbl_to_id[ann1["target_name"]],
                     "subtitle": ann1["subtitle"],
                     }
            dict2 = {"image": img2,
                     "correct_target_name": ann2["target_name"],
                     "correct_target_id": self.lbl_to_id[ann2["target_name"]],
                     "subtitle": ann2["subtitle"],
                     }

        return dict1, dict2


class TwoAugUnsupervisedTVQADataset(TVQADataset):
    def __getitem__(self, index):
        ann = self.anns[str(index)]
        dir = "./dataset/" + ann['series'] + "_frames/"
        image = Image.open(os.path.join(dir, ann['face']))

        img1, img2 = self.transform(image), self.transform(image)
        dict1, dict2 = {}, {}

        if self.split == "train":
            target = []
            if ann['name']:
                for name in ann['name']:
                    if name in self.lbl_to_id.keys():
                        target.append(self.lbl_to_id[name])
                    else:
                        target.append(self.lbl_to_id["Unknown"])
            # target = self.lbl_to_id[ann['name']]
            dict1 = {"image": img1,
                     # "weak_label": target,
                     "subtitle": ann["subtitle"],
                     # "name": ann['name'],
                     }
            dict2 = {"image": img2,
                     # "weak_label": target,
                     "subtitle": ann["subtitle"],
                     # "name": ann['name'],
                     }
        elif self.split == "test":
            dict1 = {"image": img1,
                     "correct_target_name": ann["target_name"],
                     "correct_target_id": self.lbl_to_id[ann["target_name"]],
                     "subtitle": ann["subtitle"],
                     }
            dict2 = {"image": img2,
                     "correct_target_name": ann["target_name"],
                     "correct_target_id": self.lbl_to_id[ann["target_name"]],
                     "subtitle": ann["subtitle"],
                     }

        return dict1, dict2


class BuildTVQADataset(TVQADataset):
    def __init__(self):
        self.entity_recognition_model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.label_list = [
            "O",  # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",  # Beginning of a person's name right after another person's name
            "I-PER",  # Person's name
            "B-ORG",  # Beginning of an organisation right after another organisation
            "I-ORG",  # Organisation
            "B-LOC",  # Beginning of a location right after another location
            "I-LOC"  # Location
        ]
        self.project_dir = "./"
        self.series = "friends"
        # self.all_subtitles_loc = self.project_dir + "dataset/tvqa_subtitles/"
        self.all_subtitles_loc = self.project_dir + f"dataset/hmtl/{self.series}/"
        self.season = None
        self.episode = None
        self.subtitle_json = None

        self.build_dict()
        self.anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/{self.series}_annotations.json")
        # self.split = "train"
        # self.lbl_to_id = self.load_labels(self.anns, lbl_path=self.project_dir + f"dataset/{self.series}_lbl_to_id.json")
        # self.create_hist("bbt_train_hist_plusunknowns", lbl_type="name")

        # self.build_evaluation_dict(self.project_dir + f"dataset/new_tvqa_plus_test_annotations.json")
        # dataset_dict = self.add_missing_frames_from_test_set_to_trainset(dataset_path=self.project_dir + f"dataset/new_new_dict_with_names.json")
            # dataset_dict = self.stitch_tokens_in_dict(dataset_dict)
        # save_json(dataset_dict, f"./dataset/new_stitched_new_dict_with_names.json")
        # dataset_dict = load_json(os.path.join(self.project_dir, f"dataset/new_stitched_new_dict_with_names.json"))
        # # clean_dataset_dict = self.clean_dict(dataset_dict, path=self.project_dir + f"dataset/clean_new_dict_with_names.json")
        # # new_anns = self.load_anns(dataset_dict, ann_path=self.project_dir + f"dataset/new_new_dict_with_names_{self.series}_annotations.json")
        # new_anns = self.load_anns(dataset_dict, ann_path=self.project_dir + f"dataset/__new_anns_{self.series}_annotations.json")
        # old_dict = load_json("./dataset/bbt_dict.json")
        # old_anns = self.load_anns(old_dict, ann_path=self.project_dir + f"dataset/__old_anns_{self.series}_annotations.json")
        # # old_anns = self.load_anns(None, ann_path=self.project_dir + f"dataset/{self.series}_annotations.json")
        # start = len(old_anns)
        # for ann in new_anns.values():
        #     old_anns[str(start)] = ann
        #     start += 1
        # save_json(old_anns, "./dataset/compare_with_bbt_train_annotations.json")
        # save_json(old_anns, "./dataset/new_mix_bbt_annotations.json")

    def load_srt(self, srt_dir, srt_cache_path):
        """
        return: A python dict, the keys are the video names, the entries are lists,
                each contains all the text from a .srt file
        sub_times are the start time of the sentences.
        """
        if os.path.exists(srt_cache_path):
            print("Found srt data cache, loading ...")
            return load_json(srt_cache_path)
        print("Loading srt files from %s ..." % srt_dir)
        # srt_paths = glob.glob(os.path.join(srt_dir, "friends_s01e02_seg02_clip_*.srt"))

        srt_paths = glob.glob(os.path.join(srt_dir, f"friends_s{self.season}e{self.episode}_seg0*.srt"))
        # srt_paths = glob.glob(os.path.join(srt_dir, f".srt"))
        name2sub_text = {}
        name2sub_face = {}
        name2sub_other = {}
        name2sub_time = {}
        name2sub_time_end = {}
        for i in tqdm(range(len(srt_paths))):
            subs = pysrt.open(srt_paths[i], encoding="iso-8859-1")
            if len(subs) == 0:
                subs = pysrt.open(srt_paths[i])

            text_list = []
            name_list = []
            other_list = []
            sub_time_list = []
            sub_time_list_end = []
            for j in range(len(subs)):
                cur_sub = subs[j]
                cur_str = cur_sub.text
                cur_str = "(<UNKNAME>:)" + cur_str if cur_str[0] != "(" else cur_str
                cur_str = cur_str.replace("\n", " ")
                names, other_labels = self.find_names(cur_str)
                text_list.append(cur_str)
                name_list.append(names)
                other_list.append(other_labels)
                sub_time_list.append(
                    60 * cur_sub.start.minutes + cur_sub.start.seconds + 0.001 * cur_sub.start.milliseconds)
                sub_time_list_end.append(
                    60 * cur_sub.end.minutes + cur_sub.end.seconds + 0.001 * cur_sub.end.milliseconds)

            key_str = os.path.splitext(os.path.basename(srt_paths[i]))[0]
            name2sub_text[key_str] = text_list
            name2sub_face[key_str] = name_list
            name2sub_other[key_str] = other_list
            name2sub_time[key_str] = sub_time_list
            name2sub_time_end[key_str] = sub_time_list_end
        if len(srt_paths) == 0:
            print("season and episode combination doesn't exist.")
            return
        srt_data = {"sub_text": name2sub_text,
                    "sub_face": name2sub_face,
                    "sub_other": name2sub_other,
                    "sub_time": name2sub_time,
                    "sub_time_end": name2sub_time_end}
        save_json(srt_data, srt_cache_path)
        return load_json(srt_cache_path)  # we do this because the ints will turn to string 0 -> '0'

    def load_anns(self, dataset, ann_path):

        if os.path.exists(ann_path):
            print("Found annotation cache, loading ...")
            return load_json(ann_path)

        anns = {}
        all_faces = []
        annid = 0
        for img in tqdm(dataset.keys()):
            # if there is at least one face:
            if dataset[img]['faces']:
                for face, bbox, face_landmark in zip(dataset[img]['faces'], dataset[img]['bbox'], dataset[img]['face_points']):
                    anns[annid] = {'face': face,
                                   'name': dataset[img]['names'],
                                   'img': dataset[img]['img'],
                                   'subtitle': dataset[img]['subtitle'],
                                   'clip': dataset[img]['clip'],
                                   'series': dataset[img]['series'],
                                   'bbox': bbox,
                                   'face_points': face_landmark}
                    annid += 1
                    # if dataset[img]['names']:
                    #     for name in dataset[img]['names']:
                    #         anns[annid] = {'face': face,
                    #                        'name': name,
                    #                        'img': dataset[img]['img'],
                    #                        'subtitle': dataset[img]['subtitle'],
                    #                        'clip': dataset[img]['clip'],
                    #                        'series': dataset[img]['series'],
                    #                        'bbox': bbox,
                    #                        'face_points': face_landmark}
                    #         annid += 1
                    # else:
                    #     anns[annid] = {'face': face,
                    #                    'name': "Unknown",
                    #                    'img': dataset[img]['img'],
                    #                    'subtitle': dataset[img]['subtitle'],
                    #                    'clip': dataset[img]['clip'],
                    #                    'series': dataset[img]['series'],
                    #                    'bbox': bbox,
                    #                    'face_points': face_landmark}
                    #     annid += 1
        save_json(anns, ann_path)
        return load_json(ann_path)


    def load_ann_mentions(self, dataset, ann_path):

        if os.path.exists(ann_path):
            print("Found annotation cache, loading ...")
            return load_json(ann_path)

        anns = {}
        all_faces = []
        annid = 0
        for img in tqdm(dataset.keys()):
            # if there is at least one face:
            if dataset[img]['faces']:
                for face, bbox, face_landmark in zip(dataset[img]['faces'], dataset[img]['bbox'], dataset[img]['face_points']):
                    anns[annid] = {'face': face,
                                   'name': dataset[img]['names'],
                                   'img': dataset[img]['img'],
                                   'subtitle': dataset[img]['subtitle'],
                                   'clip': dataset[img]['clip'],
                                   'series': dataset[img]['series'],
                                   'bbox': bbox,
                                   'face_points': face_landmark}
                    annid += 1
                    # if dataset[img]['names']:
                    #     for name in dataset[img]['names']:
                    #         anns[annid] = {'face': face,
                    #                        'name': name,
                    #                        'img': dataset[img]['img'],
                    #                        'subtitle': dataset[img]['subtitle'],
                    #                        'clip': dataset[img]['clip'],
                    #                        'series': dataset[img]['series'],
                    #                        'bbox': bbox,
                    #                        'face_points': face_landmark}
                    #         annid += 1
                    # else:
                    #     anns[annid] = {'face': face,
                    #                    'name': "Unknown",
                    #                    'img': dataset[img]['img'],
                    #                    'subtitle': dataset[img]['subtitle'],
                    #                    'clip': dataset[img]['clip'],
                    #                    'series': dataset[img]['series'],
                    #                    'bbox': bbox,
                    #                    'face_points': face_landmark}
                    #     annid += 1
        save_json(anns, ann_path)
        return load_json(ann_path)


    def build_dataset_mentions(self, srt_data, dataset_path):
        if srt_data is None:
            return
        series_list = ["castle", "friends", "grey", "house", "met", "bbt"]
        series_directory = self.project_dir + f"dataset/frames_hq/{series_list[1]}_frames/"
        save_directory = self.project_dir + f"/dataset/{series_list[1]}_frames/"
        all_mentions = set(load_json(self.project_dir + f"/dataset/hmtl/mentions/list_mentions_filtered_{self.series}.json"))  # todo list or set
        Path(save_directory).mkdir(parents=True, exist_ok=True)

        if os.path.exists(dataset_path):
            print("Found dataset cache, loading ...")
            return load_json(dataset_path)

        # mtcnn = MTCNN(keep_all=True)
        i = 0
        dataset_dict = {}
        for clip in tqdm(srt_data['mentions'].keys()):
            for t, mentions in enumerate(srt_data['mentions'][clip]):

                intersect = set(mentions).intersection(all_mentions)
                if intersect:
                    # print(intersect)
                    frame_num_begin = math.ceil(srt_data['sub_time'][clip][t] * 3)
                    frame_num_end = math.ceil(srt_data['sub_time_end'][clip][t] * 3)
                    clip_dir = series_directory + clip

                    for frame_num in range(frame_num_begin, frame_num_begin + 4):
                        img_str = str(frame_num).zfill(5)
                        if os.path.exists(os.path.join(clip_dir, img_str + ".jpg")):
                            # make a function to draw
                            # img = Image.open(os.path.join(clip_dir, img_str + ".jpg"))

                            dataset_dict[i] = {"series": self.series, "clip_dir": clip_dir, "clip": clip,
                                               "img": img_str,
                                               "subtitle": srt_data['sub_text'][clip][t],
                                               "mentions": list(mentions),
                                               "filtered_mentions": list(intersect),
                                               "first_frame": True if frame_num == frame_num_begin else False}
                            i += 1

        # dataset_dict = self.stitch_tokens_in_dict(dataset_dict)
        save_json(dataset_dict, dataset_path)
        return load_json(dataset_path)

    def build_dict(self):
        '''
        This function extracts all the information needed from the "tvqa dataset" and builds the annotation files
        so we could later run TVQADataset on it and load the data
        :return:
        '''
        for s in range(1, 11):
            for i in range(1, 26):
                if i < 10:
                    self.episode = f"0{i}"
                else:
                    self.episode = f"{i}"
                if s < 10:
                    self.season = f"0{s}"
                else:
                    self.season = f"{s}"
                print(f"season: {self.season}, episode: {self.episode}")

                self.subtitle_json = self.all_subtitles_loc + f"subtitle_cache_{self.series}_s{self.season}e{self.episode}.json"
                srt_data = self.load_srt(self.all_subtitles_loc, self.subtitle_json)
                # dataset_dict = self.prepare_tvqa_json(srt_data, dataset_path=self.project_dir + f"dataset/{self.series}_dict_s{self.season}e{self.episode}.json")
                dataset_dict = self.build_dataset_mentions(srt_data, dataset_path=self.project_dir + f"dataset/mentions/{self.series}_dict_s{self.season}e{self.episode}.json")
        dict_paths = glob.glob(os.path.join("./", f"dataset/mentions/{self.series}_dict_*.json"))
        stitched_dict = {}
        for i in tqdm(range(len(dict_paths))):
            start = len(stitched_dict.keys())
            dict_i = load_json(dict_paths[i])
            stitched_dict.update({start+key: value for key, value in enumerate(dict_i.values())})

        save_json(stitched_dict, f"./dataset/mentions/{self.series}_dict.json")
        dataset_dict = load_json(os.path.join(self.project_dir, f"dataset/mentions/{self.series}_dict.json"))
        #
        # # clean_dataset_dict = self.clean_dict(dataset_dict, path=self.project_dir + f"dataset/clean_{self.series}_dict.json")
        # self.anns = self.load_anns(dataset_dict, ann_path=self.project_dir + f"dataset/{self.series}_annotations.json")

    def add_missing_frames_from_test_set_to_trainset(self, dataset_path):
        if os.path.exists(dataset_path):
            return load_json(dataset_path)

        save_directory = self.project_dir + f"dataset/bbt_frames/"
        # test_anns = load_json(f"./dataset/new_tvqa_plus_test_annotations.json")
        # images_in_test = set()
        # for ann in test_anns.values():
        #     images_in_test.add(f"{ann['clip']}_{ann['img']}")
        # save_json(list(images_in_test), "./dataset/images_in_test.json")

        images_in_test = set(load_json("./dataset/images_in_test.json"))
        train_images_with_face = set(load_json("./dataset/images_with_face.json"))
        mtcnn = MTCNN(keep_all=True)
        i = 0
        new_dict = {}

        for test_img in tqdm(list(images_in_test)):
            if test_img not in train_images_with_face:
                clip, img_name = "_".join(test_img.split('_')[:-1]), test_img.split('_')[-1]
                img_path = os.path.join("./", f"dataset/frames_hq/bbt_frames/{clip}/{img_name}.jpg")

                subtitle, names = find_corresponding_subtitle(clip, int(img_name))

                img = Image.open(img_path)
                boxes, probs, points = mtcnn.detect(img, landmarks=True)
                faces = []
                if boxes is not None:
                    img_draw = img.copy()
                    draw = ImageDraw.Draw(img_draw)
                    for f, (box, point) in enumerate(zip(boxes, points)):
                        draw.rectangle(box.tolist(), width=5)
                        faces.append("{}_{}_{}.png".format(clip, img_name, f))
                        box[0] = box[0] - 20
                        box[1] = box[1] - 20
                        box[2] = box[2] + 20
                        box[3] = box[3] + 20
                        extract_face(img, box,
                                     save_path=save_directory + "{}_{}_{}.png".format(clip, img_name, f))
                if points is not None:
                    points = points.tolist()
                if boxes is not None:
                    boxes = boxes.tolist()

                new_dict[i] = {"series": "bbt",
                               "clip": clip,
                               "img": img_name,
                               "names": names,
                               "subtitle": subtitle,
                               "faces": faces,
                               "face_points": points,
                               "bbox": boxes}

                i += 1

        save_json(new_dict, dataset_path)
        return load_json(dataset_path)

    def add_missing_frames(self, dataset_path):
        save_directory = self.project_dir + f"dataset/bbt_frames/"
        images_with_face = load_json("./dataset/images_with_face.json")
        images_with_face = set(images_with_face)

        mtcnn = MTCNN(keep_all=True)
        i = 0
        new_dict = {}

        clip_paths = glob.glob(os.path.join("./", f"dataset/frames_hq/bbt_frames/*"))
        for clip_path in clip_paths:
            clip = clip_path.split('/')[-1]
            img_paths = glob.glob(os.path.join("./", f"dataset/frames_hq/bbt_frames/{clip}/*"))
            for img_path in img_paths:
                img_name = img_path.split('/')[-1]
                img_name = img_name.split('.')[0]

                if clip+'_'+img_name not in images_with_face: # we found a missing face
                    img = Image.open(img_path)
                    boxes, probs, points = mtcnn.detect(img, landmarks=True)
                    faces = []
                    if boxes is not None:
                        img_draw = img.copy()
                        draw = ImageDraw.Draw(img_draw)
                        for f, (box, point) in enumerate(zip(boxes, points)):
                            draw.rectangle(box.tolist(), width=5)
                            faces.append("{}_{}_{}.png".format(clip, img_name, f))
                            box[0] = box[0] - 20
                            box[1] = box[1] - 20
                            box[2] = box[2] + 20
                            box[3] = box[3] + 20
                            extract_face(img, box,
                                         save_path=save_directory + "{}_{}_{}.png".format(clip, img_name, f))
                    if points is not None:
                        points = points.tolist()
                    if boxes is not None:
                        boxes = boxes.tolist()
                    new_dict[i] = {"series": "bbt",
                                   "clip": clip,
                                   "img": img_name,
                                   "names": [],
                                   "subtitle": "",
                                   "faces": faces,
                                   "face_points": points,
                                   "bbox": boxes}
                    i += 1

        save_json(new_dict, dataset_path)
        return load_json(dataset_path)

    def build_evaluation_dict(self, dataset_path):
        '''
        step 1: if no subtitle, discard
        step 2: if obj['label'] in objects.json then it's an object -> discard
        step 3: iou -> not implemented here
        step 4: if obj['label'] not in fuzzy_name_matchings then it's unknown
        This function loads the annotatations of "tvqa+" dataset for evaluation
        :return:
        '''
        series_list = ["castle", "friends", "grey", "house", "met", "bbt"]
        series_directory = self.project_dir + f"dataset/frames_hq/{series_list[5]}_frames/"
        save_directory = self.project_dir + f"dataset/new_tvqa_plus_{series_list[5]}_frames/"
        objects = set(load_json(self.project_dir + f"dataset/objects.json"))

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        fuzzy_name_matchings = load_json("./dataset/fuzzy_name_matchings.json")
        main_characters = [item for sublist in fuzzy_name_matchings.values() for item in sublist]

        if os.path.exists(dataset_path):
            print("Found dataset cache, loading ...")
            return load_json(dataset_path)

        def complete_dict(tvqa_plus_dict, i, anns):
            for ann in tqdm(anns):
                clip = ann['vid_name']
                for img_id, image in ann['bbox'].items():
                    # step 1: if no subtitle, discard
                    subtitle, names = find_corresponding_subtitle(clip, int(img_id))
                    if not subtitle:
                        continue
                    for obj_id, obj in enumerate(image):
                        # step 2: if obj['label'] in objects.json then it's an object -> discard
                        if obj['label'] in objects:
                            continue
                        # step 3: if iou = 0 then we assume it's an object -> discard -> this is done later

                        # step 4: if obj['label'] not in fuzzy_name_matchings then it's unknown
                        if obj['label'] not in main_characters:
                            char_name = "Unknown"
                        else:
                            for char_name in fuzzy_name_matchings.keys():
                                if obj['label'] in fuzzy_name_matchings[char_name]:
                                    break
                        img_str = str(obj['img_id']).zfill(5)
                        clip_dir = series_directory + ann['vid_name']
                        img = Image.open(os.path.join(clip_dir, img_str + ".jpg"))
                        img = img.crop((obj['left'], obj['top'], obj['left'] + obj['width'], obj['top'] + obj['height']))
                        img.save(save_directory + "{}_{}_{}.png".format(ann['vid_name'], img_str, obj_id))
                        tvqa_plus_dict[i] = {"series": "bbt",
                                             "clip": ann['vid_name'],
                                             "subtitle": subtitle,
                                             "face": "{}_{}_{}.png".format(ann['vid_name'], img_str, obj_id),
                                             "img": img_str,
                                             "weak_lbls": names,
                                             "name": char_name, #gt_label
                                             "bbox": [obj['left'], obj['top'], obj['left'] + obj['width'],
                                                      obj['top'] + obj['height']]}
                        i += 1
            return tvqa_plus_dict, i

        tvqa_plus_dict = {}
        i = 0
        train_anns = load_json(self.project_dir + f"dataset/tvqa+/tvqa_plus_annotations/tvqa_plus_train.json")
        tvqa_plus_dict, i = complete_dict(tvqa_plus_dict, i, train_anns)

        val_anns = load_json(self.project_dir + f"dataset/tvqa+/tvqa_plus_annotations/tvqa_plus_val.json")
        tvqa_plus_dict, i = complete_dict(tvqa_plus_dict, i, val_anns)

        save_json(tvqa_plus_dict, dataset_path)
        return load_json(dataset_path)


def find_corresponding_subtitle(clip, img_number):
    srt_data = load_json(f"./dataset/tvqa_subtitles/subtitle_cache_bbt_{clip.split('_')[0]}.json")
    subtitle = ""
    for t, (begin_time, end_time) in enumerate(
            zip(srt_data['sub_time'][clip], srt_data['sub_time_end'][clip])):
        current_time = math.floor(img_number / 3)
        names = []
        if current_time >= begin_time and current_time <= end_time:
            subtitle = srt_data['sub_text'][clip][t]
            names = srt_data['sub_face'][clip][t]
            if names:
                names = [name_lbl[0] for name_lbl in names]
            break

    return subtitle, names


def fuzzy_matching():

    train = load_json("./tvqa+/tvqa_plus_annotations/tvqa_plus_train.json")
    gt_labels = []
    for ann in train:
        for image in ann['bbox'].values():
            for obj in image:
                gt_labels.append(obj['label'])

    val = load_json("./tvqa+/tvqa_plus_annotations/tvqa_plus_val.json")
    gt_labels_val = []
    for ann in val:
        for image in ann['bbox'].values():
            for obj in image:
                gt_labels_val.append(obj['label'])

    def create_hist(obj_list):
        labels = set(obj_list)
        num_classes = len(labels)
        # print(f"number of labels: {num_classes}, The unique labels are: {labels}")
        lbl_to_id = {}
        for idx, lbl in enumerate(labels):
            lbl_to_id[lbl] = idx
        hist_bins = np.arange(num_classes + 1)
        histogram = np.zeros((num_classes,), dtype=np.int)
        classes = [lbl_to_id[obj] for obj in obj_list]
        histogram += np.histogram(classes, bins=hist_bins)[0]
        ind_sorted = np.argsort(histogram)[::-1]
        id_to_lbl = {id: lbl for (lbl, id) in lbl_to_id.items()}
        class_names_sorted = [id_to_lbl[ind] for ind in ind_sorted]
        return class_names_sorted

    cls_names = create_hist(gt_labels)

    scores = torch.zeros([len(cls_names), len(cls_names)])
    for i1, n1 in enumerate(cls_names):
        for i2, n2 in enumerate(cls_names):
            scores[i1, i2] = fuzz.ratio(n1, n2)

    cls_names_pd = pd.DataFrame(cls_names, columns=['cls_names'])
    scores_df = pd.DataFrame(scores)

    fuzzy_name_matchings = {}
    # indices corresponding to these classes Sheldon, Leonard, Penny, Howard, Raj, Amy, Bernadette, Stuart
    for i, name in zip([0, 1, 2, 3, 4, 5, 6, 15],["Sheldon", "Leonard", "Penny", "Howard", "Raj", "Amy", "Bernadette", "Stuart"]):
        fuzzy_name_matchings[name] = set(cls_names_pd.loc[scores_df[i] > 70, "cls_names"])

    cls_names_val = create_hist(gt_labels_val)
    scores_val = torch.zeros([len(cls_names_val), len(cls_names_val)])
    for i1, n1 in enumerate(cls_names_val):
        for i2, n2 in enumerate(cls_names_val):
            scores_val[i1, i2] = fuzz.ratio(n1, n2)

    cls_names_val_pd = pd.DataFrame(cls_names_val, columns=['cls_names_val'])
    scores_val_df = pd.DataFrame(scores_val)

    # indices corresponding to these classes Sheldon, Leonard, Penny, Howard, Raj, Amy, Bernadette, Stuart
    for i, name in zip([0, 1, 2, 3, 4, 5, 6, 15], ["Sheldon", "Leonard", "Penny", "Howard", "Raj", "Amy", "Bernadette", "Stuart"]):
        fuzzy_name_matchings[name].update(set(cls_names_val_pd.loc[scores_val_df[0] > 70, "cls_names_val"]))

    # you need to check them by hand if they make sense
    save_json({key: list(value) for key, value in fuzzy_name_matchings.items()}, "./dataset/fuzzy_name_matchings.json")


def split_train_test_episodes():
    train_dataset = TVQADataset(series="friends", split="train", transform=get_test_transforms(series="friends"))
    test_dataset = TVQADataset(series="friends", split="test", transform=get_test_transforms(series="friends"))
    train_friends = {}
    test_friends = {}
    i, j = 0, 0
    for train_ann, test_ann in zip(train_dataset.anns.values(), test_dataset.anns.values()):
        if train_ann['clip'].split("_")[1].endswith('5'):
            test_friends[i] = {**train_ann, **test_ann}
            i += 1
        else:
            train_friends[j] = {**train_ann, **test_ann}
            j += 1
    save_json(train_friends, "./dataset/friends_train_annotations.json")
    save_json(test_friends, "./dataset/friends_test_annotations.json")


if __name__ == "__main__":
    # transform = transforms.Compose([transforms.ToTensor()])
    # transform = get_transforms()
    # dataset = TwoAugUnsupervisedTVQADataset(split="train", transform=transform)
    # dataset = TVQADataset(split="test", transform=transforms.Compose([transforms.ToTensor()]))

    # dataset = TVQADataset(series="bbt", split="train", transform=transforms.Compose([transforms.ToTensor()]))
    # mean, std = find_norm_data(dataset)
    # print(mean, std)
    # dataset = BuildTVQADataset()
    import shutil
    img_paths = glob.glob(os.path.join("./", "dataset/new_tvqa_plus_bbt_frames/*"))
    mtcnn = MTCNN(keep_all=True)
    for img_path in tqdm(img_paths):
        img = Image.open(img_path)
        try:
            boxes, probs, points = mtcnn.detect(img, landmarks=True)
            if boxes is not None:
                if len(boxes) != 1:
                    print(f"image {img_path} has {len(boxes)} faces")
                img_draw = img.copy()
                draw = ImageDraw.Draw(img_draw)
                for f, (box, point) in enumerate(zip(boxes, points)):
                    draw.rectangle(box.tolist(), width=5)
                    box[0] = box[0] - 20
                    box[1] = box[1] - 20
                    box[2] = box[2] + 20
                    box[3] = box[3] + 20
                extract_face(img, box, save_path=f"./dataset/new_tvqa_plus_bbt_frames_onlyfaces/{img_path.split('/')[-1]}")
            else:
                print(f"image{img_path} has no faces")
                shutil.copy(img_path, f"./dataset/new_tvqa_plus_bbt_frames_onlyfaces/{img_path.split('/')[-1]}")
        except Exception as e:
            print(e)
            print(f"image {img_path} is not valid")
            shutil.copy(img_path, f"./dataset/new_tvqa_plus_bbt_frames_onlyfaces/{img_path.split('/')[-1]}")
            continue



    # face_to_train_idx = {}
    # for idx, ann in train_dataset.anns.items():
    #     if ann['face'] in face_to_train_idx:
    #         print("face already existed")
    # face_to_train_idx[ann['face']] = idx


    # old = load_json("./dataset/bbt_train_annotations.json")
    # new = load_json("./dataset/compare_with_bbt_train_annotations.json")
    # for (o_i, o) in old.items():
    #     if int(o_i) <= 112672:
    #         o['name'] = new[o_i]['name']
    #     else:
    #         for (n_i, n) in new.items():
    #             if int(n_i) <= 112672:
    #                 continue
    #             if o['face'] not in ["s08e20_seg02_clip_08_00108_0.png", "s08e23_seg01_clip_00_00153_0.png",
    #                                  "s08e23_seg01_clip_00_00153_1.png", "s08e23_seg01_clip_00_00153_2.png"]:
    #                 if o['face'] == n['face']:
    #                     o['name'] = n['name']
    #                     break
    # old['114324']['name'] = []
    # old['114325']['name'] = []
    # old['114326']['name'] = []
    # print(len(old))
    # save_json(old, "./dataset/bbt_train_annotations__.json")
    # import torch
    # new_train_anns = load_json("./dataset/bbt_train_annotations.json")
    # old_embs = torch.load("./output/evaluate_bbt/model/new_bbt_face_embeddings.pt")
    # for iteration, ann in tqdm(enumerate(new_train_anns.values())):
    #     if iteration == 0:
    #         new_embs = old_embs[int(ann['old_train_indices'][0])]
    #     else:
    #         new_embs = torch.cat((new_embs, old_embs[int(ann['old_train_indices'][0])]), 0)
    # torch.save(new_embs, "./dataset/bbt_faceeee_embeddings.pt")
    # em = torch.load("./dataset/bbt_faceeee_embeddings.pt")
    # em = em.reshape((272603, 512))
    # torch.save(em, "./dataset/bbt_face_embeddings_hopefully_correct.pt")
    # splits = ["test", "dev"]
    # splits = ["train"]
    # lbl_types = ["name", "target_name"]
    # for lbl_type in lbl_types:
    #     for split in splits:
    #         tvqa = TVQADataset(lbl_type=lbl_type, split=split, episode=None, transform=transform)
    #         tvqa.create_hist(f"{split}_hist_{lbl_type}", lbl_type=lbl_type)

    # train_tvqa = TVQADataset(split="train", transform=transform)
    # train_tvqa = TVQADataset(lbl_type="name", split="train", episode=None, transform=transform)
    # train_tvqa.create_hist("train_hist", lbl_type="name")
    # dev_tvqa = TVQADataset(split="dev", transform=transform)
    # dev_tvqa.create_hist("dev_hist", lbl_type="target_name")
    # test_tvqa = TVQADataset(split="test", transform=transform)
    # test_tvqa.create_hist("test_hist", lbl_type="target_name")

    # function: copy old target label annotations into new annonation file (also had to annotate some with hand)
    # old_ann_loc = "./dataset/2/2_all_annotations.json"
    # old_anns = load_json(old_ann_loc)
    #
    # for i, new_ann in enumerate(tvqa.anns.values()):
    #     for old_ann in old_anns.values():
    #         if old_ann['face'] == new_ann['face']:
    #             new_ann["target_name"] = old_ann["target_name"]
    #             new_ann["target_id"] = old_ann["target_id"]
    #             break
    # save_json(tvqa.anns, "./dataset/blabla.json")
