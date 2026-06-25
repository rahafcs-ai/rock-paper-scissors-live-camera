# Rock Paper Scissors Live Camera Classifier

This project is a real-time hand gesture classification system using a CNN model and OpenCV.

## Classes

* Rock
* Paper
* Scissors

## Project Features

* CNN model trained using TensorFlow/Keras
* Data augmentation to improve generalization
* Real-time webcam prediction using OpenCV
* Confidence threshold to avoid unreliable predictions
* Prediction smoothing using recent frames

## Results

The model achieved high test accuracy after applying data augmentation.
The test accuracy improved significantly after adding data augmentation.

## Files

* `train_model.py`: trains the CNN model and saves it.
* `live_camera.py`: runs the real-time webcam prediction.
* `collect_data.py`: optional script for collecting custom images.
* `accuracy_plot.png`: training vs validation accuracy plot.
* `loss_plot.png`: training vs validation loss plot.
* `requirements.txt`: required Python libraries.

## How to Run
Note: The trained model file is not included because of GitHub file size limits.  
Run `python train_model.py` first to train and generate `rock_paper_scissors_model.keras`, then run `python live_camera.py`.
Install the required libraries:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run live camera prediction:

```bash
python live_camera.py
```

Press `q` to quit the camera window.
