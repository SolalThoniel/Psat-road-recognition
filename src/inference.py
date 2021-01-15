import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, resnet101
from src.load_data import image_loader, DATASET_RESOLUTION_SMALL, DATASET_RESOLUTION_MEDIUM, DATASET_RESOLUTION_LARGE,\
    CLASSES_LABELS


# images = le tableau d'images
# resolution = une des 3 constantes de résolution dans load_data.py
# resnet_layers = le nombre de layers du réseau resnet (34, 50 ou 101)
# training_epoch = l'époque d'entrainement à choisir pour le modèle
def inference(images, resolution, resnet_layers=34, training_epoch=5):

    # Put the images in a dataset
    images_transformed = []

    image_transforms = transforms.Compose([
        transforms.Pad(resolution.get('padding')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for image in images:
        images_transformed.append(image_loader(image_transforms, image))

    # Device Selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading Model
    net = None
    path = None

    if resolution == DATASET_RESOLUTION_SMALL:
        if resnet_layers == 101:
            net = resnet101()
            path = "../epochs/80x45/101/road_recognition_"+str(training_epoch)+".pth"
        elif resnet_layers == 50:
            net = resnet50()
            path = "../epochs/80x45/50/road_recognition_"+str(training_epoch)+".pth"
        else:  # resnet_layer = 34
            net = resnet34()
            path = "../epochs/80x45/34/road_recognition_"+str(training_epoch)+".pth"
    elif resolution == DATASET_RESOLUTION_MEDIUM:
        if resnet_layers == 101:
            net = resnet101()
            path = "../epochs/160x90/101/road_recognition_"+str(training_epoch)+".pth"
        elif resnet_layers == 50:
            net = resnet50()
            path = "../epochs/160x90/50/road_recognition_"+str(training_epoch)+".pth"
        else:  # resnet_layer = 34
            net = resnet34()
            path = "../epochs/160x90/34/road_recognition_"+str(training_epoch)+".pth"
    elif resolution == DATASET_RESOLUTION_LARGE:
        if resnet_layers == 101:
            net = resnet101()
            path = "../epochs/320x180/101/road_recognition_"+str(training_epoch)+".pth"
        elif resnet_layers == 50:
            net = resnet50()
            path = "../epochs/320x180/50/road_recognition_"+str(training_epoch)+".pth"
        else:  # resnet_layer = 34
            net = resnet34()
            path = "../epochs/320x180/34/road_recognition_"+str(training_epoch - 1)+".pth"

    net.load_state_dict(torch.load(path))
    net = net.to(device)

    results = []

    with torch.no_grad():
        for image in images_transformed:
            image = image.to(device)
            outputs = net(image)
            _, predicted = torch.max(outputs, 1)
            results.append(CLASSES_LABELS[predicted])

    return results
