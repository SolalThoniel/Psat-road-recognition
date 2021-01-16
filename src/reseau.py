
import torch.nn as nn
import torch.nn.functional as f

class reseau(nn.Module):
    # constructeur, p nombre de variables explicatives
    def __init__(self):
        # appel du constructeur de l'ancêtre
        super(reseau, self).__init__()
        self.features = nn.Sequential(
            #80*45 en entrée
            nn.Conv2d(3,32, kernel_size=7, stride = 1),
            #74*39
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #36*19
            nn.Conv2d(32,64, kernel_size=3, padding = 1),
            #36*19
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride = 2),
            #17*9
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride = 2)
            #8*4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*8*4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128*8*4)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = reseau()
    print(net)