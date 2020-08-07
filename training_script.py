import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import training_fns as fns
import torch.onnx
import onnx
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=25, required=True, help="number of training epochs")
args = ap.parse_args()

if __name__ == "__main__":
    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is: ", device)
    base_path = os.getcwd()
    print("current working directory is: ", base_path)
    experiment_path = base_path + "/data/"
    print("experiment working directory is: ", experiment_path)

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
        
    train_dir = os.path.join(experiment_path, "train/")
    val_dir = os.path.join(experiment_path, "val/")
    test_dir = os.path.join(experiment_path, "test/")

    print("train dir is: ", train_dir)
    print("val dir is: ", val_dir)
    print("test dir is: ", test_dir)

    train_images = {x: datasets.ImageFolder(y, data_transform) for x, y in zip(["train", "val"], [train_dir, val_dir])}
    dataloader = {x: torch.utils.data.DataLoader(train_images[x], 
                                                batch_size=16, 
                                                shuffle=True, 
                                                num_workers=4) 
                                                for x in ["train", "val"]}
    test_images = datasets.ImageFolder(test_dir, data_transform)
    test_dataloader = torch.utils.data.DataLoader(test_images, 
                                                batch_size=16, 
                                                shuffle=True, 
                                                num_workers=4)

    class_names = train_images["train"].classes
    dataset_sizes = {x: len(train_images[x]) for x in ["train", "val"]}
    
    print("Class names: {}".format(class_names))
    print("Dataset sizes -- train: {}, val: {}".format(dataset_sizes["train"], dataset_sizes["val"]))
    print("Test set size: {}".format(len(test_images)))

    # define the model
    print("Downloading model: {}".format("mobilenet_v2"))
    model_pretrained = torch.hub.load("pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True)
    print(model_pretrained.classifier)
    model_pretrained.classifier = nn.Sequential(nn.Dropout(0.2, inplace=False),
                                                nn.Linear(1280, len(class_names), bias=True))
    model_pretrained = model_pretrained.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_pretrained.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=1e-1)

    # train the model
    print("training the model")
    trained_model, losses, accs = fns.train_model(train_images,
                                                    dataloader, 
                                                    model_pretrained, 
                                                    criterion, 
                                                    optimizer, 
                                                    exp_lr_scheduler, 
                                                    num_epochs=args.epochs)
    
    # plot loss functions
    print("plotting loss functions")
    epochs = [i+1 for i, x in enumerate(losses[0])]
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, losses[0])
    plt.plot(epochs, losses[1])
    plt.title("Training Loss and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"])
    plt.savefig("loss.pdf")

    plt.figure(figsize=(10,10))
    plt.plot(epochs, accs[0])
    plt.plot(epochs, accs[1])
    plt.title("Training Accuracy and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"])
    plt.savefig("accuracy.pdf")

    # test the model
    print("testing the model")
    fns.test_model(test_dataloader, trained_model, class_names)

    print("saving the model")
    torch.save(trained_model, base_path + "final.pth.tar")

    print("saving the onnx file")
    inputs, classes = next(iter(dataloader["train"]))
    trained_model.eval()
    outputs = trained_model(inputs)
    torch.onnx.export(trained_model,                # model being run
                  inputs,                           # model input (or a tuple for multiple inputs)
                  "facemask_classifier.onnx",       # where to save the model (can be a file or file-like object)
                  export_params=True,               # store the trained parameter weights inside the model file
                  opset_version=10,                 # the ONNX version to export the model to
                  do_constant_folding=True,         # whether to execute constant folding for optimization
                  input_names = ["input"],          # the model's input names
                  output_names = ["output"],        # the model's output names
                  dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                                "output" : {0 : "batch_size"}})
    
    print("testing the onnx file")
    onnx_model = onnx.load("facemask_classifier.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("facemask_classifier.onnx")

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: fns.to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(fns.to_numpy(outputs), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
