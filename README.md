# Facebook User Predictions

This repository was made by the team Ambridge Canalytica.
It contains our solution for the project assignment of IFT6758 Fall 2019.

## Setting up

1. To run our software, you'll have to install Anaconda. Select the Python 3 version:
https://www.anaconda.com/distribution/#download-section
2. After installing anaconda, run:
    * conda env create -f environment.yaml
    * This will create a conda environment called fb-env which we will use to run this software

## Outputted predictions format

When obtaining predictions without software, we will save an `.xml` file for every person's predictions inside
`submission/predictions/<current-date>/` with contents like the following:
    ```xml
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
    ```
    where:
    * `<id>` and id is the Facebook user id
    * All the other entries are our predictions when it comes to that Facebook user

## How to obtain pre-trained predictions

**Note**: Make sure you follow the steps in **Setting up** first. 

This solution is pre-trained and is ready to serve predictions.
If you are on linux, you can simply run the following to obtain predictions using the pre-trained models:

 ```bash
./ift6758 -i <path-to-contents-of-new-data.zip> -o <path-to-save-predictions>
```

or you can run:

```bash
python ift6758.py -i <path-to-contents-of-new-data.zip> -o <path-to-save-predictions>
```

Or, in PyCharm you can:
1. Go to Run -> Edit Configurations, expand Templates on the left and select Python
2. Set the Script path to the absolute path to the ift6758.py script
3. Set the Parameters as the parameters seen above in the bash call:
    * -i <path-to-contents-of-new-data.zip> -o <path-to-save-predictions>
    * you can also use longhand versions: --input_path and --output_path

When you run the script, the output will adhere to the information in **Outputted predictions format**

### Training new estimators

**Note**: Make sure you follow the steps in **Setting up** first.

Here follow instructions on how to modify this software and retrain our estimators.

1. We recommend using the software PyCharm. You can download it here:
https://www.jetbrains.com/pycharm/download/#section=windows
2. Open the root folder of this software in PyCharm
3. Go to File -> Settings.
4. Select the project on the left in the dialog that appears and expand its properties.
5. From the list of interpreters, select the fb-env that was created in **Setting up**
6. On the left, select Project Structure.
7. Make sure the content root (on the right) is the root folder of this software
8. Click on the src folder in the middle list and click on Mark as: Source on top
9. Press ok to close the dialog

Now you have our dependencies installed and a working PyCharm setup to use our code.

1. Train a model. Take a look in src/train.py. Here you have three dictionaries:
    * age_estimators
    * gender_estimators
    * personality_estimators
2. Note which estimators specified inside each of the dictionaries above you want to use. To train using those you
can either (if you are on linux), run: 

```bash
python src/train.py --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline
```

which would train using the baseline estimator option from each estimator dictionary, or in PyCharm you could:
1. Go to Run -> Edit Configurations, expand Templates on the left and select Python
2. Set the Script path to the absolute path to the src/train.py script
3. Set the Parameters as the parameters seen above in the bash call:
    * --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline

Regardless of whether you run the bash command or this command above in PyCharm, you should have the same results.
This will:
* Load the training data from `../new_data/Train/` (the software expects the new_data to be present in the parent 
folder to itself)
* Use the baseline variant of each estimator
* Save a model named `%Y-%m-%d_%H.%M.%S.pkl` in the folder `submission/models/`.

Alternatively:
* Use `--data_path` if you want to explicitly specify where the data is loaded from.
* Use `--save_path` if you want the model saved in a different location.

Example (or use Run -> Edit Configurations in PyCharm like mentioned above):

```bash
python src/train.py --data_path="../new_data/Train/" --save_path="my-models/" --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline
```

This will:
* Load the training data from `../new_data/Train/`
* Use the baseline variant of each estimator
* Save a model named `%Y-%m-%d_%H.%M.%S.pkl` in the folder `submission/my-models/`.
   
### Obtaining predictions using your newly trained model

1. You can either:
    * Set the variable `MODEL_PATH` inside `submission/ift6768.py` to the model that was created in the step above
    (see specific instructions on what the path should be from the software standard output after it creates the model)
    * Or specify the same path with the argument --model_path when invoking the src/predict.py script. See below
2. Then, you run the prediction script:
    ```bash 
    python ift6758.py -i "../new_data/Public_Test/" -o "my-predictions/"
    ```
    Alternatively:
   * Use `-i` or `--input_path` if you want to explicitly specify where the test data is loaded from.
   * Use `-o` or `--output_path` if you want the test data predictions saved in a different location.
    Example:
    ```bash 
    python ift6758.py -i "../new_data/Public_Test/" -o "my-predictions/"
    ```

When you run the script, the output will adhere to the information in **Outputted predictions format**
