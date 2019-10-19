import os
import pathlib

from typing import List

from data.personality_traits import PersonalityTraits


class UserLabels:
    def __init__(self,
                 user_id: str,
                 age: str,
                 gender: int,
                 personality_traits: PersonalityTraits):

        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.personality_traits = personality_traits

    def save_obj(self, save_path: str) -> None:
        """
        Serializes the object data and stores it in the following format in a file save_path/user_id.xml:
        <user
        id="8157f43c71fbf53f4580fd3fc808bd29"
        age_group="xx-24"
        gender="female"
        extrovert="2.7"
        neurotic="4.55"
        agreeable="3"
        conscientious="1.9"
        open="2.1"
        />
        """
        # We could use something fancier like an xml serializer, but this is a pretty simple XML blob.
        # I think just string formatting should be fine.
        predicted_values = {
            'userid': self.user_id,
            'age_group': self.age,
            'gender': "female" if self.gender == 1 else "male",
            'extrovert': self.personality_traits.extroversion,
            'neurotic': self.personality_traits.neuroticism,
            'agreeable': self.personality_traits.agreeableness,
            'conscientious': self.personality_traits.conscientiousness,
            'open': self.personality_traits.openness
        }
        xml_blob = """
        <user 
        id="{userid}"
        age_group="{age_group}"
        gender="{gender}"
        extrovert="{extrovert}"
        neurotic="{neurotic}"
        agreeable="{agreeable}"
        conscientious="{conscientious}"
        open="{open}"
        />
        """.format(**predicted_values).strip()

        save_file_path = os.path.join(
            save_path,
            "{}.xml".format(self.user_id)
        )
        with open(save_file_path, "w") as f:
            f.write(xml_blob)

    @staticmethod
    def save(predictions: List['UserLabels'], save_path: str) -> str:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        for prediction in predictions:
            prediction.save_obj(save_path)
        return save_path

    def __repr__(self):
        return """
        user_id: {} \n
        age: {} \n
        gender: {} \n
        personality_traits: {}
        """.format(self.user_id, self.age, self.gender, self.personality_traits)
