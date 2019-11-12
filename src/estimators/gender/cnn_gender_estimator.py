from typing import List
import pandas as pd

import torch
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data.dataloader import DataLoader

from data.image_dataset import ImageDataset
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from estimators.base.gender_estimator import GenderEstimator
from networks.cnn import BasicNet
from util.utils import read_config_file


class CnnGenderEstimator(GenderEstimator):

    def __init__(self):
        config = read_config_file("src/config/configGender.json")
        if config["model"] == "BasicNet":
            self.neural_net = BasicNet()
        self.valid_split = config["valid_split"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.learning_rate = config["learning_rate"]
        self.max_epochs = config["max_epochs"]
        self.prediction: int = None

    def fit(
        self,
        features: List[UserFeatures],
        oxford_df: pd.DataFrame,
        labels: List[UserLabels]
    ) -> None:

        train_features, train_labels, valid_features, valid_labels = self.train_valid_split(
            features,
            labels,
            valid_split=self.valid_split
        )

        train_dataloader = DataLoader(
            dataset=ImageDataset(train_features, train_labels),
            batch_size=self.train_batch_size,
            shuffle=True
        )

        valid_dataloader = DataLoader(
            dataset=ImageDataset(valid_features, valid_labels),
            batch_size=self.train_batch_size,
            shuffle=True
        )

        trainer = create_supervised_trainer(
            model=self.neural_net,
            optimizer=torch.optim.Adam(self.neural_net.parameters(), self.learning_rate),
            loss_fn=torch.nn.BCELoss()
        )

        evaluator = create_supervised_evaluator(
            model=self.neural_net,
            metrics={
                'accuracy': Accuracy(),
                'BCE': Loss(torch.nn.BCELoss())
            }
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_dataloader)
            metrics = evaluator.state.metrics
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(trainer.state.epoch, metrics['accuracy'], metrics['BCE']))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(valid_dataloader)
            metrics = evaluator.state.metrics

            print("Validation Results - Epoch[{}] Avg accuracy: {:.2f}".format(
                trainer.state.epoch,
                metrics['accuracy']
            ))

        trainer.run(train_dataloader, max_epochs=self.max_epochs)

    def predict(
        self,
        features: List[UserFeatures],
        oxford_df: pd.DataFrame
    ) -> List[int]:

        test_dataloader = DataLoader(
            dataset=ImageDataset(features, labels=None),
            batch_size=self.test_batch_size,
            shuffle=True
        )

        self.prediction = []

        for batch_idx, (data) in enumerate(test_dataloader):
            self.prediction.append(self.neural_net(data))

        self.prediction = torch.stack(self.prediction, 0).detach().numpy()
        self.prediction = self.prediction.reshape(-1, 2)
        self.prediction = self.prediction.argmax(axis=1)

        return self.prediction.tolist()
