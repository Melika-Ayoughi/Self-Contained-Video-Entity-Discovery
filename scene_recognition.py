import vissl
import tensorboard
import apex
import torch
import json
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from PIL import Image
import torchvision.transforms as transforms
import glob, os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
# All other options override the simclr_8node_resnet.yaml config.

cfg = [
    'config=benchmark/linear_image_classification/places205/models/regnet32Gf.yaml',
    'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=./content/regnet_seer.torch', # Specify path for the model weights.
    'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
    'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk.
    'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
    'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=True', # Do not flatten features.
    'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5", ["Identity", []]]]' # Extract only the res5avg features.
]

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)



model = build_model(cfg.MODEL, cfg.OPTIMIZER)
# Load the checkpoint weights.
weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)


# Initializei the model with the simclr model weights.
model = init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)

print("Weights have loaded")


def extract_features(sampling_rate=None, batch_size=10):
    # episode ='01'
    # imgs_dir = glob.glob(f"./dataset/frames_hq/friends_frames/friends_s01e{episode}_seg0*")
    imgs_dir = glob.glob(f"./dataset/frames_hq/bbt_frames/*")
    i = 0
    batch = 0
    first_batch = True
    for img_dir in tqdm(imgs_dir):
        imgs_path = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img in imgs_path:
            if i % sampling_rate != 0:
                i += 1
                continue
            i += 1
            image = Image.open(img)
            # Convert images to RGB. This is important
            # as the model was trained on RGB images.
            image = image.convert("RGB")
            # Image transformation pipeline.
            pipeline = transforms.Compose([
              transforms.Resize(size=(224, 224)),
              # transforms.CenterCrop(224),
              transforms.ToTensor(),
            ])
            x = pipeline(image)
            x = x.unsqueeze(0)
            if batch == 0:
                img_batch = x
            else:
                img_batch = torch.cat((img_batch, x), 0)
            batch += 1
            if batch == batch_size:
                print(f"extracting features for i {i} . . .")
                features = model(img_batch)
                # features = features[0]
                if first_batch:
                    all_embeddings = features[0]
                    print(f"size of features: {features[0].shape}")
                    first_batch = False
                else:
                    all_embeddings = torch.cat((all_embeddings, features[0]), 0)
                batch = 0

    if batch != batch_size: #the last batch
        features = model(img_batch)
        # features = features[0]
        all_embeddings = torch.cat((all_embeddings, features[0]), 0)

    # print("extracting features . . .")
    # features = model(img_batch)
    # features = features[0]
    # print(f"Features extracted have the shape: { features.shape }")
    torch.save(all_embeddings, f"./dataset/bbt_scene_embeddings_rate_{sampling_rate}.pt")
    return all_embeddings


def save_json(data, file_path):
    with open(file_path, "w+") as f:
        json.dump(data, f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_embeddings():
    dataset_dict = load_json(f"./dataset/mentions/friends_dict.json")

    first_img = True
    for i, data in enumerate(dataset_dict.values()):
        if data['first_frame']:
            image = Image.open(os.path.join(data['clip_dir'], data['img'] + ".jpg"))
            image = image.convert("RGB")
            pipeline = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ])
            x = pipeline(image)
            x = x.unsqueeze(0)
            features = model(x)
            dataset_dict[str(i)]['embedding'] = features[0]
            dataset_dict[str(i+1)]['embedding'] = features[0]
            dataset_dict[str(i+2)]['embedding'] = features[0]
            dataset_dict[str(i+3)]['embedding'] = features[0]

            if first_img:
                all_embeddings = features[0]
                first_img = False
            else:
                all_embeddings = torch.cat((all_embeddings, features[0]), 0)

            # dataset_dict[i] = {"series": self.series, "clip_dir": clip_dir, "clip": clip,
            #                    "img": img_str,
            #                    "subtitle": srt_data['sub_text'][clip][t],
            #                    "mentions": list(mentions),
            #                    "filtered_mentions": list(intersect),
            #                    "first_frame": True if frame_num == frame_num_begin else False}

    torch.save(all_embeddings, f"./dataset/friends_scenes_with_mention_embs.pt")
    return all_embeddings


def visualize_tsne(tsne_grid, image_loader=None):

    # num_classes = len(id_to_lbl)
    # convert to pandas
    # label_ids = pd.DataFrame(label_ids, columns=['label'])['label']
    # create a scatter plot.
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # if not label_ids.isnull().values.any():
    #     plt.scatter(tsne_grid[:, 0], tsne_grid[:, 1], lw=0, s=40, c=np.asarray(label_ids),
    #                 cmap=discrete_cmap(num_classes, "tab10"))
    #     # , c = palette[np.asarray([lbl_to_id[lbl] for lbl in colors])]
    #     # c = np.random.randint(num_classes, size=len(tsne_grid[:, 1]))
    # else:
    #     plt.scatter(tsne_grid[:, 0], tsne_grid[:, 1], lw=0, s=40)
    plt.xlim(-114, 178)
    plt.ylim(-117, 268)
    # cbar = plt.colorbar(ticks=range(num_classes))
    # cbar.set_ticklabels(list(id_to_lbl.values()))
    # plt.clim(-0.5, num_classes - 0.5)
    # ax.axis('off')
    # ax.axis('tight')
    dataset_dict = load_json(f"./dataset/mentions/friends_dict.json")

    max_dim = 16
    max_x, max_y, min_x, min_y = 0, 0, 100, 100
    for i, data in enumerate(dataset_dict.values()):
        if data['first_frame']:
            x,y = tsne_grid[i//4]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
            tile = Image.open(os.path.join(data['clip_dir'], data['img'] + ".jpg"))
            # data['embedding']
            # tile = image_loader.get(os.path.join(data['clip_dir'], data['img'] + ".jpg"))
            # tile = Image.open(img)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            imagebox = OffsetImage(tile) #, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), pad=0.1)
            ax.add_artist(ab)
    print(f"max x: {max_x}, min x {min_x}, max y {max_y}, min y {min_y}")
    return fig, plt

    # max_dim = 16
    # imgs = glob.glob('./dataset/frames_hq/friends_frames/friends_s01e01_seg02_clip_17/*.jpg')
    # for i, ((x, y), img) in enumerate(zip(tsne_grid, imgs)):
    #     # print(i, x, y, img)
    #     # tile = image_loader.get(img)
    #     tile = Image.open(img)
    #     rs = max(1, tile.width/max_dim, tile.height/max_dim)
    #     tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    #     imagebox = OffsetImage(tile) #, zoom=0.2)
    #     ab = AnnotationBbox(imagebox, (x, y), pad=0.1)
    #     ax.add_artist(ab)
    #
    # return fig, plt

    # max_dim = 16
    # imgs_dir = glob.glob(f"./dataset/frames_hq/bbt_frames/*")
    # i = 0
    # tsne_counter = 0
    # for img_dir in tqdm(imgs_dir):
    #     imgs_path = glob.glob(os.path.join(img_dir, "*.jpg"))
    #     for img in imgs_path:
    #         if i % sampling_rate != 0:
    #             i += 1
    #             continue
    #         i += 1
    #         tile = Image.open(img)
    #         x, y = tsne_grid[tsne_counter]
    #         # print(x, y)
    #         rs = max(1, tile.width / max_dim, tile.height / max_dim)
    #         tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
    #         imagebox = OffsetImage(tile)  # , zoom=0.2)
    #         ab = AnnotationBbox(imagebox, (x, y), pad=0.1)
    #         ax.add_artist(ab)
    #         tsne_counter += 1
    #
    # return fig, plt


# get_embeddings()

all_embeddings = torch.load(f"./dataset/friends_scenes_with_mention_embs.pt")
tsne_grid = TSNE(random_state=10, n_iter=4000).fit_transform(all_embeddings.detach().numpy())
fig, plt = visualize_tsne(tsne_grid)
fig.savefig(os.path.join(f"./output/scene_recognition/", f"friends_scene.pdf"))
plt.clf()

# sampling_rate = 1000
# # scene_embeddings = extract_features(sampling_rate=sampling_rate, batch_size=10)
# # scene_embeddings = torch.load("./dataset/scene_embeddings_all.pt")
# scene_embeddings = torch.load(f"./dataset/bbt_scene_embeddings_rate_{sampling_rate}.pt")
# tsne_grid = TSNE(random_state=10, n_iter=4000).fit_transform(scene_embeddings.detach().numpy())
# fig, plt = visualize_tsne(tsne_grid)
# fig.savefig(os.path.join(f"./output/scene_recognition/", f"bbt_scene_rate_{sampling_rate}.pdf"))
# plt.clf()


# sampling_rate = 100
# scene_embeddings = extract_features(sampling_rate=sampling_rate, batch_size=100)
# # scene_embeddings = torch.load("./dataset/scene_embeddings_all.pt")
# # scene_embeddings = torch.load(f"./dataset/bbt_scene_embeddings_rate_{sampling_rate}.pt")
# tsne_grid = TSNE(random_state=10, n_iter=2000).fit_transform(scene_embeddings.detach().numpy())
# fig, plt = visualize_tsne(tsne_grid) #size of features:
# fig.savefig(os.path.join(f"./output/scene_recognition/", f"bbt_scene_rate_{sampling_rate}.pdf"))
# plt.clf()