import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, resnet50, resnet101
from src import load_data


def run():
    # Hyperparameters
    num_epochs = 10
    batch_size = 10
    learning_rate = 0.01
    momentum = 0.9
    step_size = 5
    gamma = 0.1

    # Device Selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading Data
    data_loader = load_data.Data(batch_size=batch_size)

    # Loading Model
    net = resnet50()
    net.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training Loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training
        for i, data in enumerate(data_loader.train_loader, 0):
            # Data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Print Statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Validation
        inference(data_loader, net, device, mode='Validation')

        # Test
        inference(data_loader, net, device, mode='Test')

        scheduler.step()

        path = '../epochs/road_recognition_' + str(epoch) + '.pth'
        torch.save(net.state_dict(), path)

    print('Finished Training')


def inference(data_class, net, device, mode='Validation'):
    if mode == 'Validation':
        data_loader = data_class.valid_loader
    else:
        data_loader = data_class.test_loader

    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1

    print('Accuracy of the network on the %d %s images: %d %%'
          % (sum(class_total), mode, 100 * sum(class_correct) / sum(class_total)))

    print("Nb of Piste-Cyclable : " + str(class_total[0]))
    print("Nb of Route : " + str(class_total[1]))
    print("Nb of Sentier : " + str(class_total[2]))
    print("Nb of Trottoir : " + str(class_total[3]))
    print("Nb of Voie-Partagee : " + str(class_total[4]))

    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
            data_class.classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    run()

