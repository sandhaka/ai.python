import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from time import time

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainSet = datasets.MNIST('./data/vision/handwritten_digits/', download=True, train=True, transform=transform)
valSet = datasets.MNIST('./data/vision/handwritten_digits/', download=True, train=False, transform=transform)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
valLoader = torch.utils.data.DataLoader(valSet, batch_size=64, shuffle=True)

dataiter = iter(trainLoader)

# Uncomment to Debug ---

# import matplotlib.pyplot as plt

# images, labels = dataiter.next()
# print(type(images))
# print(images.shape)
# print(labels.shape)
#
# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
# plt.show()
#
# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
#
# plt.show()

# ------------

# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} ...")
model.to(device)

criterion = nn.NLLLoss()
images, labels = next(iter(trainLoader))
images = images.view(images.shape[0], -1)

logps = model(images.cuda())
loss = criterion(logps, labels.cuda())

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainLoader:

        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images.cuda())
        loss = criterion(output, labels.cuda())

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainLoader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

correct_count, all_count = 0, 0
for images, labels in valLoader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(img.cuda())

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

torch.save(model, './data/vision/handwritten_digits/trained/model.pt')

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
