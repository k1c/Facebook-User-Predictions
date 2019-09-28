# Setup

## Train and Evaluate on Server
* Enter the main project directory
```bash
cd Facebook-User-Predictions
```
* Train the baseline model. This will save a model called (`fb_estimator_[hex_id]`) to the current directory. If you want it in a different location specify the `save_path` argument. 
```python
python src/train.py --data_path="../Train" --age_estimator=baseline --gender_estimator=baseline --personality_estimator=baseline
```

* Test the saved model.
```python
python ift6758.py -i [full_path_to_trained_model] -o [full_path_to_save_dir]
```



