from time import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from ._base import BaseExperiment
from dataset.Dataset import DataBowl2018Dataset
from utils.load_functions import load_model
from utils.plot_functions import plot_test_image
from utils.calculate_metrics import metrics

class DataBowl2018Experiment(BaseExperiment) :
    def __init__(self, device, data_type, dataset_dir, train_frame, val_frame, test_frame, image_size, batch_size, num_workers,
                 model_name, epochs, optimizer, criterion, lr, momentum, weight_decay,
                 pretrained_encoder, step, train, c_fold, save_path):
        self.data_type = data_type
        self.image_size = image_size
        self.pretrained_encoder = pretrained_encoder
        self.num_channels = 3 if pretrained_encoder else 1
        self.step = step
        self.train = train
        self.c_fold = c_fold
        self.save_path = save_path

        num_classes = 1

        train_transform, train_target_transform = self.transform_generator('train')
        test_transform, test_target_transform = self.transform_generator('test')

        train_data = DataBowl2018Dataset(dataset_dir, train_frame, transform=train_transform, target_transform=train_target_transform)
        val_data = DataBowl2018Dataset(dataset_dir, val_frame, transform=test_transform, target_transform=test_target_transform)
        test_data = DataBowl2018Dataset(dataset_dir, test_frame, transform=test_transform, target_transform=test_target_transform)

        self.train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers = num_workers, pin_memory=True,
                                       worker_init_fn=self.seed_worker)
        self.val_loader = DataLoader(val_data, batch_size = 1, shuffle=False, num_workers = num_workers, pin_memory=True,
                                       worker_init_fn=self.seed_worker)
        self.test_loader = DataLoader(test_data, batch_size = 1, shuffle=False, num_workers = num_workers, pin_memory=True,
                                       worker_init_fn=self.seed_worker)

        super(DataBowl2018Experiment, self).__init__(device, model_name, num_classes, self.num_channels, optimizer, criterion,
                                                     epochs, batch_size, lr, momentum, weight_decay, image_size, self.train_loader)

        self.history_generator()

    def fit(self):
        print("start experiment!!")
        self.print_params()

        if not self.train :
            print("Inference")
            # load model
            self.model = load_model(self.save_path, self.data_type, self.image_size,
                                    self.batch_size, self.model_name, self.lr, self.epochs, self.c_fold)
            test_loss = self.inference()

            return test_loss
        else :
            for epoch in tqdm(range(1, self.epochs + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.epochs))
                if epoch % 10 == 0:
                    self.print_params()

                epoch_start_time = time()
                print("TRAINING")
                train_loss = self.train_epoch(epoch)

                print("VALIDATING")
                val_loss = self.val_epoch(epoch)

                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                print('\nEpoch {}/{} : train loss {}  | val_loss {} |current lr {} | took {} h {} m {} s'.format(
                    epoch, self.epochs, train_loss, val_loss, self.current_lr(self.optimizer), int(h), int(m), int(s)))

            return self.model, self.optimizer, self.history

    def train_epoch(self, epoch):
        self.model.train()
        running_loss, total = 0., 0

        for batch_idx, (image, target) in enumerate(self.train_loader) :
            loss = self.forward(image, target)
            self.backward(loss)

            running_loss += loss.item()
            total += image.size(0)

            if (batch_idx + 1) % self.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    running_loss / total
                ))

        return running_loss / total

    def val_epoch(self, epoch):
        self.model.eval()
        total_loss, total = .0, 0

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.val_loader):
                if (batch_idx + 1) % self.step == 0:
                    print("{}/{}({}%) COMPLETE".format(
                        batch_idx + 1, len(self.val_loader), np.round((batch_idx + 1) / len(self.val_loader) * 100), 4))

                loss = self.forward(image, target)
                total_loss += loss.item()
                total += target.size(0)

        val_loss = total_loss / total

        if (batch_idx + 1) % self.step == 0 or (batch_idx + 1) == len(self.val_loader):
            print("Epoch {} | test loss : {}".format(epoch, val_loss))

        return val_loss

    def inference(self):
        self.model.eval()
        total_loss, total = .0, 0

        with torch.no_grad() :
            accuracies, precisions, recalls, f1_scores, ious, aps = [], [], [], [], [], []
            for batch_idx, (image, target) in enumerate(self.test_loader) :
                if (batch_idx + 1) % self.step == 0:
                    print("{}/{}({}%) COMPLETE".format(
                        batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                image, target = image.to(self.device).float(), target.to(self.device)

                # loss = self.forward(image, target)
                if self.pretrained_encoder :
                    image = image.repeat(1, 3, 1, 1)

                if self.model_name == 'fdsnet' :
                    output = self.model(image, train=False)
                else :
                    output = self.model(image)
                predict = torch.sigmoid(output)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                acc, pre, rec, f1, iou, ap = metrics(target, predict)
                accuracies.append(acc)
                precisions.append(pre)
                recalls.append(rec)
                f1_scores.append(f1)
                ious.append(iou)
                aps.append(ap)
                total += target.size(0)

                if self.c_fold == 0 :
                    plot_test_image(image, target, predict, total, self.data_type, self.model_name)


        test_loss = total_loss / total
        accuracy = np.mean(accuracies)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1_score = np.mean(f1_scores)
        iou = np.mean(ious)
        ap = np.mean(aps)

        return test_loss, accuracy, precision, recall, f1_score, iou, ap

    def history_generator(self):
        self.history = dict()
        self.history['train_loss'] = list()
        self.history['val_loss'] = list()

    def transform_generator(self, mode):
        if mode == 'train' :
            transform_list = [
                transforms.RandomCrop(self.image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5]),
            ]

            target_transform_list = [
                transforms.RandomCrop(self.image_size),
                transforms.ToTensor(),
            ]

        else :
            transform_list = [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5]),
            ]

            target_transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),

            ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)