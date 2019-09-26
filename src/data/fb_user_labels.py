from data.personality_traits import PersonalityTraits


class FBUserLabels:
    def __init__(self,
                 user_id: str,
                 age: str,
                 gender: int,
                 personality_traits: PersonalityTraits):

        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.personality_traits = personality_traits

    @classmethod
    def from_data(cls,
                  user_id: str,
                  age: str,
                  gender: int,
                  personality_traits: PersonalityTraits) -> 'FBUserLabels':
        return cls(
            user_id=user_id,
            age=age,
            gender=gender,
            personality_traits=personality_traits
        )

    def save(self, save_path: str):
        """
        Serializes the object data and stores it in the following format in a file save_path/user_id.xml:
        <userid="8157f43c71fbf53f4580fd3fc808bd29"
        age_group="xx-24"
        gender="female"
        extrovert="2.7"
        neurotic="4.55"
        agreeable="3"
        conscientious="1.9"
        open="2.1"
        />
        """
        pass

    def __repr__(self):
        return """
        user_id: {} \n
        age: {} \n
        gender: {} \n
        personality_traits: {}
        """.format(self.user_id, self.age, self.gender, self.personality_traits)
