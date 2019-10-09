from typing import List, Dict

import torch
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy
from torch import nn
from torch.utils.data.dataloader import DataLoader

from data.fb_image_dataset import FBImageDataset
from data.fb_user_features import FBUserFeatures
from data.fb_user_labels import FBUserLabels
from estimators.base.gender_estimator import GenderEstimator


class CnnGenderEstimator(GenderEstimator):

    def __init__(self, neural_net: nn.Module, config: Dict):
        self.neural_net = neural_net
        self.valid_split = config["valid_split"]
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]

    def fit(self, features: List[FBUserFeatures], labels: List[FBUserLabels]):

        train_features, train_labels, valid_features, valid_labels = self.train_valid_split(
            features,
            labels,
            valid_split=self.valid_split
        )

        train_dataloader = DataLoader(
            dataset=FBImageDataset(train_features, train_labels),
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_dataloader = DataLoader(
            dataset=FBImageDataset(valid_features, valid_labels),
            batch_size=self.batch_size,
            shuffle=True
        )

        trainer = create_supervised_trainer(
            model=self.neural_net,
            optimizer=torch.optim.Adam(self.neural_net.parameters()),
            loss_fn=torch.nn.BCELoss()
        )

        evaluator = create_supervised_evaluator(
            model=self.neural_net,
            metrics={
                'accuracy': Accuracy()
            }
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(valid_dataloader)
            metrics = evaluator.state.metrics
            print("Validation Results - Epoch[{}] Avg accuracy: {:.2f}".format(
                trainer.state.epoch,
                metrics['accuracy']
            ))

        trainer.run(train_dataloader, max_epochs=self.max_epochs)

    def predict(self, features: List[FBUserFeatures]) -> List[int]:
        pass
