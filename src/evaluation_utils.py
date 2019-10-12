from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

from data.personality_traits import PersonalityTraits


def regression_score(predicted: List[PersonalityTraits], true: List[PersonalityTraits]) -> List[float]:
    result = list()
    predicted_arr = np.array([x.as_list() for x in predicted])
    true_arr = np.array([x.as_list() for x in true])
    for column in range(true_arr.shape[1]):
        result.append(
            sqrt(
                mean_squared_error(true_arr[:, column], predicted_arr[:, column])
            )
        )
    return result
