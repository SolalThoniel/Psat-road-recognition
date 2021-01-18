
import torch.nn as nn
import torch.nn.functional as f

class reseau2(nn.Module):
    # constructeur, p nombre de variables explicatives
    def __init__(self):
        # appel du constructeur de l'ancêtre
        super(reseau2, self).__init__()
        self.features = nn.Sequential(
            #80*45 en entrée
            nn.Conv2d(3,32, kernel_size=(9,9)),
            #72*37
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=(1,2)),
            #35*35
            nn.Conv2d(32,64, kernel_size=5),
            #31*31
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride = 2),
            #15*15
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride = 2)
            #6*6
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 5),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), 128*6*6)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = reseau2()
    print(net)