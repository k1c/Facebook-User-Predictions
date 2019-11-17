from typing import List

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data.dataloader import DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.fb_relation_v1_preprocessed_dataset import FBRelationV1PreprocessedDataset
from estimators.base.age_estimator import AgeEstimator
from networks.nn import BasicNN


class RelationV1PersonalityEstimator(AgeEstimator):
    def __init__(self):
        self.neural_net = BasicNN(2, [8, 16, 8], 5)
        self.batch_size = 10
        self.learning_rate = 0.01
        self.max_epochs = 100
        self.predictions = []

    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            np.array([feature.likes_preprocessed_v1 for feature in features]).reshape(-1, 2),
            np.array([
                np.array(label.personality_traits.as_list())
                for label in labels
            ]),
            train_size=0.8,
            shuffle=True
        )
        x_train = Tensor(x_train).float()
        x_test = Tensor(x_test).float()
        y_train = Tensor(y_train).float()
        y_test = Tensor(y_test).float()

        train_data_loader = DataLoader(
            dataset=FBRelationV1PreprocessedDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_data_loader = DataLoader(
            dataset=FBRelationV1PreprocessedDataset(x_test, y_test),
            batch_size=self.batch_size,
            shuffle=True
        )

        trainer = create_supervised_trainer(
            model=self.neural_net,
            optimizer=torch.optim.Adam(self.neural_net.parameters(), self.learning_rate),
            loss_fn=torch.nn.MSELoss()
        )

        evaluator = create_supervised_evaluator(
            model=self.neural_net,
            metrics={
                'MSE': Loss(torch.nn.MSELoss())
            }
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_data_loader)
            metrics = evaluator.state.metrics
            print("Training Results - Epoch: {}. Avg MSE loss: {:.8f}"
                  .format(trainer.state.epoch, metrics['MSE']))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(valid_data_loader)
            metrics = evaluator.state.metrics

            print("Validation Results - Epoch {}. Avg MSE loss: {:.8f}".format(
                trainer.state.epoch,
                metrics['MSE']
            ))

        trainer.run(train_data_loader, max_epochs=self.max_epochs)

    def predict(self, features: List[UserFeatures]) -> List[str]:
        features = np.array([feature.likes_preprocessed_v1 for feature in features]).reshape(-1, 2)

        test_data_loader = DataLoader(
            dataset=FBRelationV1PreprocessedDataset(features, None),
            batch_size=self.batch_size,
            shuffle=True
        )

        for batch_idx, (data) in enumerate(test_data_loader):
            output = self.neural_net(data)
            self.predictions.append(output)

        return self.predictions
