import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator


# --------------------------------------------------------------------------------

class ChexnetTrainer():

    # ---- Train the densenet network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- nnClassCount - number of output classes
    # ---- trBatchSize - batch size
    # ---- trMaxEpoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    # ---- checkpoint - if not None loads the model and continues training

    def train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
              trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint, model_save_dir=None):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        # if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()

        model = torchvision.models.resnet18(pretrained=True)
        num_classes = nnClassCount
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

        model = model.to('cuda')
        # model = torch.nn.DataParallel(model, device_ids=[0,]).cuda()

        # -------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        # transformList.append(normalize)
        transformSequence = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS

        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
                                        transform=transformSequence)
        datasetVal = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal,
                                      transform=transformSequence)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=2,
                                     pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=2,
                                   pin_memory=True)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        # optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer = optim.Adam(model.parameters(), lr=5*10e-5)#, weight_decay=1e-5)
        optimizer = optim.Adam(model.parameters(), lr=10e-4, weight_decay=1e-5)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)#, dampening=0.5, weight_decay=0.01, nesterov=False)
        # scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
        scheduler = None

        # -------------------- SETTINGS: LOSS
        # loss = torch.nn.BCELoss(size_average = True)

        samples_lens = [22287, 9648]  # set specific class_weights
        # samples_lens=[1000, 500]  #  set specific class_weights
        samples_lens = np.array(samples_lens)
        samples_max = np.max(samples_lens)
        samples_lens = [samples_max / d for d in samples_lens]
        class_weights = torch.FloatTensor(samples_lens).to('cuda')

        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.BCEWithLogitsLoss()

        # ---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # ---- TRAIN THE NETWORK

        acc_best = 0
        for epochID in range(0, trMaxEpoch):

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            lossTrain, lr = ChexnetTrainer.epochTrain(model, dataLoaderTrain, optimizer, scheduler, criterion)
            # lossVal = ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            print(f"Loss in epoch {epochID} :::: lossTrain: {lossTrain} with lr: {lr}")
            lossVal, acc = ChexnetTrainer.evaluate(model, dataLoaderVal, criterion, all_class=True, return_numpy=False)
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            # scheduler.step(losstensor.data[0])
            # scheduler.step(lossVal)

            if acc > acc_best:
                # if False:
                acc_best = acc
                model_save_dir = os.path.join(os.path.abspath('.'), model_save_dir)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                torch.save({'epoch': epochID, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           os.path.join(model_save_dir, 'epoch-' + str(epochID + 1) + '.pth.tar'))

    # --------------------------------------------------------------------------------

    def epochTrain(model, dataLoader, optimizer, scheduler, criterion):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # training with either cpu or cuda
        model = model.to(device=device)  # to send the model for training on either cuda or cpu

        loss_ep = 0
        model.train()
        for batch_idx, (data, targets) in enumerate(dataLoader):
            # print(batch_idx)
            data = data.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            scores = model(data)
            # loss = criterion(scores,targets)
            # loss = criterion(scores,targets.squeeze(1).long())    # for CrossEntropyLoss input in segmentation
            loss = criterion(scores.squeeze(), targets.squeeze())  # for BCE
            loss.backward()
            optimizer.step()
            # scores = torch.argmax(scores, dim=1)  # evaluate
            # print(data.shape, targets.shape, scores.shape)
            if scheduler != None:
                scheduler.step()
            loss_ep += loss.item()
            # torch.cuda.empty_cache()
        loss_value = loss_ep / len(dataLoader);
        lr = optimizer.param_groups[0]['lr']

        return loss_value, lr

    # --------------------------------------------------------------------------------

    def epochVal(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        model.eval()
        loss_ep = 0
        for i, (input, target) in enumerate(dataLoader):
            input = input.to('cuda')
            target = target.to('cuda')
            with torch.no_grad():
                scores = model(input)
                losstensor = loss(scores, target.squeeze(1).long())
                loss_ep += losstensor.item()

        return loss_ep / len(dataLoader)

    def evaluate(modelTest, test_loader, criterion, all_class=False, return_numpy=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # training with either cpu or cuda
        modelTest = modelTest.to(device=device)
        modelTest.eval()

        loss_ep = 0
        targetsList = []
        predictionsList = []
        import datetime
        starttime = datetime.datetime.now()
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = modelTest(data)
                # loss = criterion(scores, targets.squeeze(1).long())    # for CrossEntropyLoss input
                loss = criterion(scores.squeeze(), targets.squeeze())  # for BCE
                loss_ep += loss.item()
                # predictions = scores.argmax(1)
                predictions = scores  # probs
                targetsList.append(targets.to('cpu'))
                predictionsList.append(predictions.to('cpu'))
            endtime = datetime.datetime.now()
            seconds = (endtime - starttime).seconds

            targets = torch.cat(targetsList, dim=0).flatten()
            predictions = torch.cat(predictionsList, dim=0).flatten()
            if return_numpy:
                return targets.numpy(), predictions.numpy()  # return same pointer
            auc = roc_auc_score(targets.numpy(), predictions.numpy())
            print(f"        auc: {auc}")
            return loss, auc
            # cmpare=(predictions==targets)
            # num_correct=cmpare.sum()
            # num_samples=predictions.shape[0]

            # loss=loss_ep / len(test_loader)
            # acc=float(num_correct) / float(num_samples)
            # # print(
            # #     f"{modelTest._get_name()} Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f} using {seconds} sec"
            # # )
            # print(f"                     lossVal: {loss} ")
            # print(f"        acc: {acc}")
            # if all_class:
            #     labels=['N', 'Y']
            #     acc_dic={}
            #     for c in range(len(labels)): acc_dic[c]=0
            #     for i in range(targets.shape[0]):
            #         t=int(targets[i])
            #         # acc_dic[t]=acc_dic[t]+1 if cmpare[i] else 0   # ×
            #         acc_dic[t]=acc_dic[t]+(1 if cmpare[i] else 0)     # √
            #     targets=targets.tolist()
            #     for c in range(len(labels)): print("%12d" % c, end=" ")
            #     print(' ')
            #     for c in range(len(labels)): print("%12.3f" %( acc_dic[c]/targets.count(c) ), end=" ")
            #     print(' ')
            # return loss, acc

    # --------------------------------------------------------------------------------

    # ---- Computes area under ROC curve
    # ---- dataGT - ground truth data
    # ---- dataPRED - predicted data
    # ---- classCount - number of classes

    def computeAUROC(dataGT, dataPRED, classCount):
        return roc_auc_score(dataGT, dataPRED)
        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

        return outAUROC

    def bootstrap_ci(labels, probs, percentile=True):
        """
        labels: list of list of labels
        probs : list of list of probabilities
        output: confidence interval of the model
        """
        import random
        labels = np.array([tensor.numpy() for tensor in labels])
        probs = np.array([tensor.numpy() for tensor in probs])
        # standardize if labels/probs was just a list
        if len(probs.shape) == 1:
            labels = labels[np.newaxis, :]
            probs = probs[np.newaxis, :]

        n_examples = probs.shape[1]
        runs = probs.shape[0]
        aucs = []

        print(labels.shape)
        print(probs.shape)
        for i in range(runs):
            for j in range(len(labels[0])):
                assert (labels[0, j] == labels[i, j]), f"{i}, {j} not equal"

        # https://www.hindawi.com/journals/jps/2015/934362/
        # for each one of our resamples:
        for i in range(10000):
            # get a set of resampling indices
            idxs = [random.randint(0, n_examples - 1) for i in range(n_examples)]
            # resample using the random resampling
            resampled_probs = probs[:, idxs]
            resampled_labels = labels[:, idxs]

            runs_auc = []
            # for each run, grab the auc score
            for j in range(runs):
                try:
                    auc = roc_auc_score(resampled_labels[j, :],
                                        resampled_probs[j, :])
                except Exception:
                    auc = 0.5
                runs_auc.append(auc)
            # take the average AUC
            auc = np.mean(runs_auc)
            aucs.append(auc)

        mean_auc = np.mean(aucs)

        bot_percentile = np.percentile(aucs, 2.5, interpolation="nearest")
        top_percentile = np.percentile(aucs, 97.5, interpolation="nearest")
        return mean_auc, bot_percentile, top_percentile

    def evaluate2(m, loader, mini_bsz):
        """evaluation loop"""
        device = 'cuda'
        # set model to evaluation mode (e.g., turn off dropout)
        m.eval()
        # and without updating the gradients
        # initialize the list of labels and predictions
        all_labels = torch.Tensor([])  # tensor with size [0]
        all_outputs = torch.Tensor([])  # tensor with size [0]
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device).type(torch.float).squeeze()
            labels = labels.to(device).type(torch.float).squeeze()
            # run the model using the validation inputs
            # mini_bsz = configuration["test_mini_batch_size"]

            torch.cuda.empty_cache()
            if mini_bsz:
                for i in range(0, inputs.size()[0], mini_bsz):
                    # run the model
                    print(inputs.size())
                    outputs = m(inputs[i:i + mini_bsz, :, :]).squeeze()
                    outputs = torch.sigmoid(outputs)
                    # get the loss and backprop
                    outputs = outputs.detach().cpu().flatten()
                    mini_labels = labels[i:i + mini_bsz].squeeze().flatten()
                    mini_labels = mini_labels.detach().cpu()
                    all_outputs = torch.cat((all_outputs, outputs), 0).detach()
                    all_labels = torch.cat((all_labels, mini_labels), 0).detach()
            else:
                outputs = m(inputs)
                # run a sigmoid since we're looking to binary classify
                outputs = torch.sigmoid(outputs)

                outputs = outputs.detach().cpu().squeeze()
                labels = labels.detach().cpu().squeeze()

                outputs, labels = outputs.flatten(), labels.flatten()

                # append the labels and the outputs to the proper list
                all_outputs = torch.cat((all_outputs, outputs), 0)
                all_labels = torch.cat((all_labels, labels), 0)

            # Evaluate area under ROC curve based on the ground truth
            # label and predicted probability
            try:
                eval_auc = roc_auc_score(all_labels, all_outputs)
            except Exception:
                eval_auc = 0.5
        return eval_auc, {"all_labels": all_labels,
                          "all_outputs": all_outputs}

    # --------------------------------------------------------------------------------

    # ---- Test the trained network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- nnClassCount - number of output classes
    # ---- trBatchSize - batch size
    # ---- trMaxEpoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    # ---- checkpoint - if not None loads the model and continues training

    def test(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
             trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        # if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()

        model = torchvision.models.resnet18(pretrained=True)
        num_classes = nnClassCount
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

        model = model.to('cuda')
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        # model = torch.nn.DataParallel(model, device_ids=[0,]).cuda()

        # -------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        # transformList.append(normalize)
        transformSequence = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetVal = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal,
                                      transform=transformSequence)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=2,
                                   pin_memory=True)

        criterion = torch.nn.CrossEntropyLoss()

        # lossVal, acc = ChexnetTrainer.evaluate(model, dataLoaderVal, criterion, all_class=True, return_numpy=False)
        # targets, predictions = ChexnetTrainer.evaluate(model, dataLoaderVal, criterion, return_numpy=True)
        # outAUROC = ChexnetTrainer.computeAUROC(targets, predictions, nnClassCount)
        # print("outAUROC:")
        # print(outAUROC)
        auc, output = ChexnetTrainer.evaluate2(model, dataLoaderVal, False)
        labels, probs = output['all_labels'], output['all_outputs']
        mean_auc, b_p, t_p = ChexnetTrainer.bootstrap_ci(labels, probs)
        print(f"auc:        {auc}")
        print(f"mean_auc:   {mean_auc}")
        print(f"b_p_t_p:    {b_p}, {t_p}")
# --------------------------------------------------------------------------------
