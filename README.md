# Setup

## Train and Evaluate on Server
* Enter the main project directory
```bash
cd Facebook-User-Predictions
```
* Train the baseline model. This will save a model called (`fb_estimator_[hex_id]`) to the current directory. If you want it in a different location specify the `save_path` argument. 
```bash
python src/train.py --data_path="../Train" --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline
```

* Sadly we need to hardcode the model path inside `ift6768.py`. Open up `ift6758.py` and update the `MODEL_PATH` variable with the full path of the trained model that is saved in the above step.
Now we can test.
```bash
python ift6758.py -i "../Public_Test" -o [full_path_to_save_dir]
```



