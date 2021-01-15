import cv2
#import sys
#import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, resnet101
from src.load_data import image_loader, DATASET_RESOLUTION_SMALL, DATASET_RESOLUTION_MEDIUM, DATASET_RESOLUTION_LARGE,\
    CLASSES_LABELS
from src.inference import inference
from PIL import Image
import torch.nn.functional as F


def video_to_frames(video, resolution=DATASET_RESOLUTION_MEDIUM, resnet_layers=34, path_output_dir="../video_test/out"):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = resnet101()
    # path = "../epochs/160x90/resnet101/road_recognition_5.pth"
    # net.load_state_dict(torch.load(path, map_location='cpu'))
    # net.to(device)
    # images_transformed = []
    # image_transforms = transforms.Compose([
    #     transforms.Pad(DATASET_RESOLUTION_MEDIUM.get('padding')),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    images = []

    vidcap = cv2.VideoCapture(video)
    count = 0
    num = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            if (count % 30) == 0:
                image = Image.fromarray(image)

                if resolution == DATASET_RESOLUTION_SMALL:
                    image = image.resize((80, 45))
                elif resolution == DATASET_RESOLUTION_MEDIUM:
                    image = image.resize((160, 90))
                elif resolution == DATASET_RESOLUTION_LARGE:
                    image = image.resize((320, 180))

                images.append(image)
                # data = image_loader(image_transforms, image)
                # print(data.size())
                # data = F.interpolate(data[0,:,:,:], size=(3,160,90))
                # data = np.resize(data[0,:,:,:], (3,160,90))

                # data = data.resize((1,3,160, 90))
                # outputs = net(data)
                # print(outputs)
                # _, predicted = torch.max(outputs, 1)
                # print(CLASSES_LABELS[predicted])
                # print(count)
                # print(inference([Image.fromarray(image)], DATASET_RESOLUTION_MEDIUM, 101))
                # cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                num = num + 1
            count += 1
        else:
            break

    cv2.destroyAllWindows()
    vidcap.release()

    print("taille images : " + str(len(images)))
    print("taille image[0] : " + str(images[0].size))

    training_epoch = 9
    results = inference(images, resolution, resnet_layers, training_epoch)

    # Stats results
    print("taille results : " + str(len(results)))
    print(results)

    piste_cyclable = 0
    route = 0
    sentier = 0
    trottoir = 0
    voie_partagee = 0

    for result in results:
        if result == "piste-cyclable":
            piste_cyclable += 1
        elif result == "route":
            route += 1
        elif result == "sentier":
            sentier += 1
        elif result == "trottoir":
            trottoir += 1
        elif result == "voie_partagee":
            voie_partagee += 1

    print("nb piste-cyclable : " + str(piste_cyclable))
    print("nb route : " + str(route))
    print("nb sentier : " + str(sentier))
    print("nb trottoir : " + str(trottoir))
    print("nb voie_partagee : " + str(voie_partagee))


def edit_map(map_path,BBox):
    df = pd.read_csv("../data/coord/coord.txt")
    print(df.head())
    map = plt.imread(map_path)
    #fig, ax = plt.subplots(figsize=(2, 2))
    fig, ax = plt.subplots()
    #ax.scatter(df.long, df.lat, zorder=1, alpha=1, c='b', s=10)
    for i in range(1,len(df.long)):
        if df.type[i]==1:
            color = "red"
        if df.type[i]==2:
            color = "blue"
        ax.plot([df.long[i-1],df.long[i]], [df.lat[i-1],df.lat[i]],linewidth=5, c=color)
    ax.set_title('Cartographie des types de routes utilisées par des cyclistes')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(map, zorder=0, extent=BBox, aspect='equal')
    plt.show()


if __name__=="__main__":
    video = 'C:/Users/solal/PycharmProjects/Psat-road-recognition/video_test/Chloe201906181022.avi'
    #video_meta_data('../data/Ambre201906031831.avi')
    video_to_frames(video, DATASET_RESOLUTION_MEDIUM, 50)

    #edit_map("../data/map.png", [4.8598,4.8884,45.7748, 45.7859])
