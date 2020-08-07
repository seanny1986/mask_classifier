import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import shutil
import time
import copy
import ssl

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None: plt.title(title)
    plt.show()

def train_model(image_datasets, dataloader, model, criterion, optimizer, scheduler, num_epochs=20, save_path=None):
    since = time.time()
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.
            running_corrects = 0.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=-1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            if phase == "train": train_loss.append(epoch_loss); train_acc.append(epoch_acc)
            else: val_loss.append(epoch_loss); val_acc.append(epoch_acc)
            print("{} loss: {:.4f} acc:{:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                if save_path is not None: print("saving best model"); torch.save(best_model, save_path + "best.pth.tar")
            print()
    time_elapsed = time.time() - since
    print("Training complete in {:0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val acc: {:4f}".format(best_acc))
    model.load_state_dict(best_model)
    return model, (train_loss, val_loss), (train_acc, val_acc)

def test_model(dataloader, model, class_names):
    model.eval()
    print("predictions")
    print("-" * 10)
    n = 1
    correct = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=-1)
        for j in range(inputs.size()[0]):
            print("true: {} predicted: {}".format(class_names[labels[j]], class_names[preds[j]]))
            if (class_names[labels[j]] == class_names[preds[j]]): correct += 1
            n += 1
    print("final accuracy is: {:4f}".format(correct/n))
    model.train()