from typing import List

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from torch.utils.data.dataloader import DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.fb_relation_v1_preprocessed_dataset import FBRelationV1PreprocessedDataset
from estimators.base.age_estimator import AgeEstimator
from networks.nn import BasicNN
from data.readers import age_category_to_int
from data.readers import int_category_to_age
from data.pre_processors import pre_process_likes_v1


class RelationV1AgeEstimator(AgeEstimator):
    def __init__(self):
        self.neural_net = BasicNN(2, 32, 4, 'classification')
        self.batch_size = 10
        self.learning_rate = 0.01
        self.max_epochs = 100

    def fit(self, features: List[UserFeatures], labels: List[UserLabels]) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            pre_process_likes_v1(features),
            np.array([
                # Converting an age category to one-hot. Example: '25-34' -> 1 -> [0, 1, 0, 0]
                np.eye(4)[np.array([age_category_to_int(label.age)])].tolist()[0]
                for label in labels
            ]),
            train_size=0.8,
            shuffle=True,
            random_state=8
        )

        x_train = torch.Tensor(x_train).float()
        x_test = torch.Tensor(x_test).float()
        y_train = torch.Tensor(y_train).float()
        y_test = torch.Tensor(y_test).float()

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
            print("Training set - Epoch: {}. Loss function AVG MSE loss: {:.8f}".format(
                trainer.state.epoch,
                metrics['MSE']
            ))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(valid_data_loader)
            metrics = evaluator.state.metrics

            print("Validation set - Epoch: {}. Loss function AVG MSE loss: {:.8f}".format(
                trainer.state.epoch,
                metrics['MSE']
            ))

        trainer.run(train_data_loader, max_epochs=self.max_epochs)

        print("Accuracy of trained model on test set: {}".format(self._get_model_accuracy(x_test, y_test)))

    def _get_predictions(self, features: torch.Tensor) -> List[str]:
        predictions = []
        for features_vector in features:
            output = self.neural_net(features_vector)
            prediction = int_category_to_age(int(torch.max(np.exp(output.detach()), 0).indices))
            predictions.append(prediction)
        return predictions

    def _get_model_accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        predictions = self._get_predictions(features)
        labels_str = [int_category_to_age(int(torch.max(label, 0)[1])) for label in labels]
        num_correct_predictions = sum(a == b for a, b in zip(predictions, labels_str))
        return num_correct_predictions / features.shape[0]

    def predict(self, features: List[UserFeatures]) -> List[str]:
        preprocessed_features = pre_process_likes_v1(features)
        return self._get_predictions(preprocessed_features)
