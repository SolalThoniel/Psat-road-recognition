import cv2
import sys
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.inference import inference


def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    num = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            if (count % 30) == 0:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                num = num + 1
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

def edit_map(map_path,BBox):
    df = pd.read_csv("../data/coord/coord.txt")
    print(df.head())
    map = plt.imread(map_path)
    fig, ax = plt.subplots(figsize=(8, 7))
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
    #video_to_frames('../data/big_buck_bunny_720p_5mb.mp4', '../data/out')
    edit_map("../data/map.png", [4.8598,4.8884,45.7748, 45.7859])
