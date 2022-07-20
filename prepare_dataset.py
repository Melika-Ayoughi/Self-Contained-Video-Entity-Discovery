import torch
import os
from torchvision import datasets
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
import random

mtcnn = MTCNN(keep_all=True)
clip_name = sorted(os.listdir(os.getcwd()))


for i in clip_name:
        clip_dir = os.path.join(os.getcwd(),i)
        for image in sorted(os.listdir(clip_dir)):
                img = Image.open(os.path.join(clip_dir, image))
                img_id = random.randint(1,10000)
                boxes, probs, points = mtcnn.detect(img, landmarks=True)
                img_draw = img.copy()
                draw = ImageDraw.Draw(img_draw)
                if boxes is None:
                    continue
                for f, (box, point) in enumerate(zip(boxes, points)):
                    draw.rectangle(box.tolist(), width=5)
                    for p in point:
                        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
                        face = extract_face(img, box, save_path='/home/mayoughi/outputs/detected_face_{}_{}.png'.format(img_id, f))
                img_draw.save('/home/mayoughi/outputs/annotated_faces_{}.png'.format(img_id))
        break
