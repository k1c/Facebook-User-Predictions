# Facebook User Predictions

This repository was made by the team Ambridge Canalytica.
It contains our solution for the project assignment of IFT6758 Fall 2019.

## Linux server information

Our linux server is located at 35.208.187.181 and our user name is user09.

Its folder structure is:
* `new_data.zip`
* `new_data/`
    * The contents of data.zip.
* `submission/`
    * The contents of this gitlab repository and where the TAs will automatically collect our software from.

## Using our software

Make sure to ssh onto our linux server mentioned above.

* Note to the TAs: Our shell script activates the base conda environment and then uses pip to install any missing
packages we need for our software from requirements.txt. Get predictions using our software with:
     ```bash
    ./ift6758 -i <path-to-contents-of-data.zip> -o <path-to-save-predictions>
    ```

### Training

1. ssh onto our linux server.
2. Enter the folder with our software, sync to latest and load and activate our custom fb-env:
    ```bash
    cd submission
    git pull
    module load anaconda/3
    conda activate fb-env
    ```
3. Train the baseline model. As an example, using the baseline estimators: 
    ```bash
    python src/train.py --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline
    ```
   This will:
   * Load the training data from `../new_data/Train/`
   * Save a model named `%Y-%m-%d_%H.%M.%S.pkl` in the folder `submission/models/`.
   
   Alternatively:
   * Use `--data_path` if you want to explicitly specify where the data is loaded from.
   * Use `--save_path` if you want the model saved in a different location.
   Example:
    ```bash
    python src/train.py --data_path="../new_data/Train/" --save_path="my-models/" --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline
    ```
   This will:
   * Load the training data from `../new_data/Train/`
   * Save a model named `%Y-%m-%d_%H.%M.%S.pkl` in the folder `submission/my-models/`.
   

### Obtaining predictions

Instructions:
1. ssh onto our linux server
2. Make sure the variable `MODEL_PATH` inside `submission/ift6768.py` references a valid trained model on
our linux machine. Follow the steps in the section "Setting up the weekly model" below for info
3. Enter the folder with our software, sync to latest and load and activate anaconda:
    ```bash
    cd submission
    git pull
    module load anaconda/3
    conda activate fb-env
    ```
4. The minimal command to obtain predictions using the configured trained model is:
    ```bash 
    python ift6758.py
    ```
   which will save an `.xml` file for every person's predictions inside `submission/predictions/<current-date>/`
   with contents like the following:
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
    
    Alternatively:
   * Use `-i` if you want to explicitly specify where the test data is loaded from.
   * Use `-o` if you want the test data predictions saved in a different location.
    Example:
    ```bash 
    python ift6758.py -i "../new_data/Public_Test/" -o "my-predictions/"
    ```
   which will save an `.xml` file for every person's predictions directly inside `submission/my-predictions/`

### Setting up the weekly model

In order for the TAs to be able to run weekly predictions, we must always have a stable version
of this repo inside the `submission/` folder along with a trained model.

Here are the steps to generate and set up a newly trained model:
1. Train a model using the "Training" section above. No need to specify the `--save_path` argument.
   Our software will print in the console where it saves the model.
2. Open `submission/ift6768.py`. and update the `MODEL_PATH` variable with the path of
the newly trained model as printed by the program in the console.
