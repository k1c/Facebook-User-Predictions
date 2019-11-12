from typing import List
import numpy as np
import pandas as pd

from data.user_features import UserFeatures
from data.user_labels import UserLabels
from data.personality_traits import PersonalityTraits
from estimators.base.personality_estimator import PersonalityEstimator

import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import math
from random import shuffle

# hyper-parameters
BATCH_SIZE = 6
NUM_TRAIN_EPOCHS = 1000
LEARNING_RATE = 1e-3
MAX_SEQ_LENGTH = 512


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class BertRegressionPersonalityEstimator(PersonalityEstimator):
    def __init__(self):
        self.encoding = BertModel.from_pretrained('bert-base-cased', output_hidden_states=False)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.regressor = torch.nn.Linear(768, 5) #bert embedding size x number of regressions
        self.optimizer = torch.optim.Adam(self.regressor.parameters(), lr=LEARNING_RATE)

    def get_statuses_with_personality_labels(self, features, labels):
        # batch-size X number of personalities bs X 5
        personality_traits = list()
        for label in labels:
            personality_traits.append(label.personality_traits.as_list())

        # batch-size X number of statuses bs X 2
        statuses = list()
        for feature in features:
            statuses.append(feature.statuses)

        dataset = list()
        for i, row in enumerate(statuses):
            for status in row:
                dataset.append((status, personality_traits[i]))

        return dataset

    # Bert is a model with absolute position embeddings so it's usually advised
    # to pad the inputs on the right rather than the left.
    # batch is a list of tensors
    def get_zero_pad(self, batch, MAX_SEQ_LENGTH):
        max_length = min(max(s.shape[1] for s in batch), MAX_SEQ_LENGTH)
        padded_batch = np.zeros((len(batch), max_length))
        for i, s in enumerate(batch):
            padded_batch[i:s.shape[1]] = s[:max_length]
        return torch.from_numpy(padded_batch).long()

    # Mask to avoid performing attention on padding token indices.
    # Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    # returns torch.FloatTensor of shape (BATCH_SIZE, sequence_length)
    def get_attention_mask(self, zero_pad_input_ids):
        attention_mask = zero_pad_input_ids.ne(0).float()  # everything in input not equal to 0 will get 1, else 0
        return attention_mask

    def fit(
        self,
        features: List[UserFeatures],
        liwc_df: pd.DataFrame,
        nrc_df: pd.DataFrame,
        labels: List[UserLabels]
    ) -> None:
        dataset = self.get_statuses_with_personality_labels(features, labels)

        #shuffle the dataset
        shuffle(dataset)

        statuses, labels = zip(*dataset)

        #cast from tuple to list
        statuses = list(statuses) #size of status is number of users * len(statuses) for each user (if each user has 2 statuses then its 100)
        labels = list(labels)

        assert len(statuses) == len(labels), "There should be an equal amount of statuses and labels (each status has one label)"

        labels = torch.Tensor(labels)

        input_ids = list() #list of torch tensors
        for status in statuses:
            input_ids.append(torch.Tensor([self.tokenizer.encode(status, add_special_tokens=True)]))

        num_batches = math.ceil(len(input_ids) / BATCH_SIZE)

        criterion = RMSELoss()

        # right now, using bert as a feature extractor and learning at linear layer level
        # if we want to fine-tune BERT, need to put the encoding parameters + regressor parameters in a list and send it to optimizer

        self.encoding.train()

        for epoch in range(NUM_TRAIN_EPOCHS):
            print("Running Training \n")
            running_loss = 0.0
            for batch_idx in range(num_batches):
                inpud_ids_batch = input_ids[batch_idx * BATCH_SIZE:(batch_idx+1) * BATCH_SIZE]
                zero_pad_input_ids_batch = self.get_zero_pad(inpud_ids_batch, MAX_SEQ_LENGTH)
                attention_mask = self.get_attention_mask(zero_pad_input_ids_batch)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.encoding(input_ids=zero_pad_input_ids_batch, attention_mask=attention_mask) # outputs is a tuple
                last_hidden_states = outputs[0]

                # last_hidden_states is of size (BATCH_SIZE, sequence_length, hidden_size)
                # and I need to bring it down to (BATCH_SIZE, hidden_size) to get a sentence representation (not a word representation)
                # therefore I can use the CLS tokens or I can average over the sequence length (chose the latter)
                sent_emb = last_hidden_states.mean(1) # (BATCH_SIZE, hidden_size)
                y_hat = self.regressor(sent_emb) # BATCH_SIZE X 5
                labels_batch = labels[batch_idx * BATCH_SIZE:(batch_idx+1) * BATCH_SIZE]
                tr_loss = criterion(y_hat, labels_batch)
                tr_loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += tr_loss.item()
                if batch_idx % 2 == 2-1:  # print every 2 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, batch_idx + 1, running_loss / 2))
                    running_loss = 0.0

    def predict(
        self,
        features: List[UserFeatures],
        liwc_df: pd.DataFrame,
        nrc_df: pd.DataFrame
    ) -> List[PersonalityTraits]:
        statuses = list()
        for feature in features:
            statuses.append(feature.statuses)

        user_input_ids = list() #list of torch tensors
        for row in statuses:
            input_ids = list()
            for status in row:
                input_ids.append(torch.Tensor([self.tokenizer.encode(status, add_special_tokens=True)]))
            user_input_ids.append(input_ids)

        predictions = list()
        with torch.no_grad():
            self.encoding.eval()
            for input_ids in user_input_ids: #for list of statuses for each user
                zero_pad_input_ids_user = self.get_zero_pad(input_ids,MAX_SEQ_LENGTH)
                attention_mask = self.get_attention_mask(zero_pad_input_ids_user)

                #forward
                outputs = self.encoding(input_ids=zero_pad_input_ids_user, attention_mask=attention_mask) # outputs is a tuple
                last_hidden_states = outputs[0]
                sent_emb = last_hidden_states.mean(1) # (status_list_length, hidden_size)
                y_hat = self.regressor(sent_emb)
                user_prediction = y_hat.mean(0) # status_list_length X 5 therefore need to average over axis 0, shape 1X5
                predictions.append(user_prediction)

        return [
            PersonalityTraits(
                openness=prediction[0].item(),
                conscientiousness=prediction[1].item(),
                extroversion=prediction[2].item(),
                agreeableness=prediction[3].item(),
                neuroticism=prediction[4].item()
            )
            for prediction in predictions
        ]
