
import torch.nn as nn
import torch.nn.functional as f

class reseau3(nn.Module):
    # constructeur, p nombre de variables explicatives
    def __init__(self):
        # appel du constructeur de l'ancêtre
        super(reseau3, self).__init__()
        self.features = nn.Sequential(
            # 80*30 en entrée
            nn.Conv2d(3, 32, kernel_size=(5)),
            # 76*26
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=(2)),
            # 37*12
            nn.Conv2d(32, 64, kernel_size=3, padding=(1, 0)),
            # 35*12
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=(1, 2)),
            # 17*10
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2)
            # 8*4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), 128*8*4)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = reseau3()
    print(net)