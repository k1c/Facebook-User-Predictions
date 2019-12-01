from typing import List

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from torch.utils.data.dataloader import DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from data.personality_traits import PersonalityTraits
from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.fb_relation_v1_preprocessed_dataset import FBRelationV1PreprocessedDataset
from estimators.base.personality_estimator import PersonalityEstimator
from networks.nn import BasicNN
from data.pre_processors import pre_process_likes_v1
from sklearn.metrics import mean_squared_error
from math import sqrt


class RelationV1PersonalityEstimator(PersonalityEstimator):
    def __init__(self):
        self.neural_net = BasicNN(2, 32, 5, 'regression')
        self.batch_size = 10
        self.learning_rate = 0.01
        self.max_epochs = 100

    def fit(
        self,
        features: List[UserFeatures],
        liwc_df: pd.DataFrame,
        nrc_df: pd.DataFrame,
        labels: List[UserLabels]
    ) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            pre_process_likes_v1(features),
            np.array([
                np.array(label.personality_traits.as_list())
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
            print("Training set - Epoch: {}. Loss function Avg MSE loss: {:.8f}".format(
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

        model_rmse_dict = self._get_model_rmse(x_test, y_test)
        print(
            "RMSE for personality regression tasks of trained model on test set: \n{}\n{}\n{}\n{}\n{}".format(
                "Openness: {}".format(model_rmse_dict['ope']),
                "Conscientiousness: {}".format(model_rmse_dict['con']),
                "Extroversion: {}".format(model_rmse_dict['ext']),
                "Agreeableness: {}".format(model_rmse_dict['agr']),
                "Neuroticism: {}".format(model_rmse_dict['neu'])
            )
        )

    def _get_predictions(self, features: torch.Tensor) -> List[PersonalityTraits]:
        predictions = np.array([
            self.neural_net(features_vector)
            for features_vector in features
        ])
        return [
            PersonalityTraits(
                openness=predictions[i][0],
                conscientiousness=predictions[i][1],
                extroversion=predictions[i][2],
                agreeableness=predictions[i][3],
                neuroticism=predictions[i][4]
            )
            for i in range(features.shape[0])
        ]

    def _get_model_rmse(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        predictions = self._get_predictions(features)
        return {
            'ope': sqrt(mean_squared_error([predictions[i].openness for i in range(features.shape[0])], labels[:, 0])),
            'con': sqrt(mean_squared_error([predictions[i].conscientiousness for i in range(features.shape[0])], labels[:, 1])),
            'ext': sqrt(mean_squared_error([predictions[i].extroversion for i in range(features.shape[0])], labels[:, 2])),
            'agr': sqrt(mean_squared_error([predictions[i].agreeableness for i in range(features.shape[0])], labels[:, 3])),
            'neu': sqrt(mean_squared_error([predictions[i].neuroticism for i in range(features.shape[0])], labels[:, 4]))
        }

    def predict(
        self,
        features: List[UserFeatures],
        liwc_df: pd.DataFrame,
        nrc_df: pd.DataFrame
    ) -> List[PersonalityTraits]:
        preprocessed_features = pre_process_likes_v1(features)
        return self._get_predictions(preprocessed_features)
