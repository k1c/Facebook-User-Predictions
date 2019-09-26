from dataclasses import dataclass


@dataclass
class PersonalityTraits:
    openness: float
    conscientiousness: float
    extroversion: float
    agreeableness: float
    neuroticism: float

    def as_list(self):
        return [
            self.openness,
            self.conscientiousness,
            self.extroversion,
            self.agreeableness,
            self.neuroticism
        ]
