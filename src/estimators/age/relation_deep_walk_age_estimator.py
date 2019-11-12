from typing import List

import torch
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from sklearn.model_selection import train_test_split
import numpy as np

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.relation_deep_walk_dataset import RelationDeepWalkDataset
from estimators.base.age_estimator import AgeEstimator
from networks.nn import BasicNN
from data.readers import age_category_to_int
from data.readers import int_category_to_age
from data.pre_processors import get_deep_walk_embeddings


class RelationDeepWalkAgeEstimator(AgeEstimator):
    def __init__(self):
        self.neural_net = BasicNN(64, 64, 4)
        self.batch_size = 10
        self.learning_rate = 0.001
        self.max_epochs = 100
        self.predictions = []

    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        user_features_embeddings = get_deep_walk_embeddings(features)
        x_train, x_test, y_train, y_test = train_test_split(
            user_features_embeddings,
            np.array([age_category_to_int(label.age) for label in labels]),
            train_size=0.8,
            shuffle=True
        )
        x_train = Tensor(x_train).float()
        x_test = Tensor(x_test).float()
        y_train = Tensor(y_train).long()
        y_test = Tensor(y_test).long()
        loss_fn = torch.nn.NLLLoss()

        train_data_loader = DataLoader(
            dataset=RelationDeepWalkDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_data_loader = DataLoader(
            dataset=RelationDeepWalkDataset(x_test, y_test),
            batch_size=self.batch_size,
            shuffle=True
        )

        trainer = create_supervised_trainer(
            model=self.neural_net,
            optimizer=torch.optim.SGD(self.neural_net.parameters(), self.learning_rate),
            loss_fn=loss_fn
        )

        evaluator = create_supervised_evaluator(
            model=self.neural_net,
            metrics={
                'Loss': Loss(loss_fn),
                'Accuracy': Accuracy()
            }
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_data_loader)
            metrics = evaluator.state.metrics
            print(
                "Training Results - Epoch: {}. Avg negative log likelihood loss: {:.8f}, Accuracy: {:.8f}".format(
                    trainer.state.epoch,
                    metrics['Loss'],
                    metrics['Accuracy']
                )
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(valid_data_loader)
            metrics = evaluator.state.metrics

            print(
                "Validation Results - Epoch: {}. Avg negative log likelihood loss: {:.8f}, Accuracy: {:.8f}".format(
                    trainer.state.epoch,
                    metrics['Loss'],
                    metrics['Accuracy']
                )
            )

        trainer.run(train_data_loader, max_epochs=self.max_epochs)

    def predict(self, features: List[UserFeatures]) -> List[str]:
        user_features_embeddings = torch.Tensor(get_deep_walk_embeddings(features))
        self.neural_net.eval()
        outputs = self.neural_net.forward(user_features_embeddings)
        probability_outputs = np.exp(outputs.detach().numpy())
        return [int_category_to_age(np.argmax(output)) for output in probability_outputs]
