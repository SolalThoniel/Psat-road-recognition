
import torch.nn as nn
import torch.nn.functional as f

class reseau2(nn.Module):
    # constructeur, p nombre de variables explicatives
    def __init__(self):
        # appel du constructeur de l'ancêtre
        super(reseau2, self).__init__()
        self.features = nn.Sequential(
            #160*90 en entrée
            nn.Conv2d(3,32, kernel_size=(9,9)),
            #152*82
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=(3)),
            #50*27
            nn.Conv2d(32,64, kernel_size=5),
            #46*23
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride = 2),
            #22*11
            nn.Conv2d(64, 128, kernel_size=3),
            #20*9
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride = 2)
            #4*9
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*4*9, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), 128*9*4)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = reseau2()
    print(net)