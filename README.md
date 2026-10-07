# Gender Classification from Images

A deep learning project for **gender classification from human images**.
The model is trained to classify an input image into one of two classes:

* **Men**
* **Women**

The project also includes a real-time webcam application that uses the trained model to make predictions from a camera feed.

## Project Overview

This project demonstrates an image classification workflow using a human-image dataset from Kaggle.

The main steps of the project are:

1. Load the image dataset.
2. Preprocess and normalize the images.
3. Prepare the data for deep learning.
4. Train an image classification model.
5. Evaluate the model.
6. Save the trained model as `model.pkl`.
7. Use the trained model for real-time webcam prediction.

## Dataset

The project uses the following Kaggle dataset:

**Human Images Dataset - Men and Women**

Dataset:
https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women

The dataset contains human images organized into two classes:

```text
men
women
```

The dataset is used to train a binary image classification model.

> **Note:** The dataset itself is not included in this repository. Please download it directly from Kaggle before running the training notebook.

## Repository Structure

```text
GenderClassification/
│
├── Untitled.ipynb     # Model training and data processing
├── camera.py          # Real-time webcam gender classification
├── model.pkl          # Trained machine learning/deep learning model
└── README.md          # Project documentation
```

## Technologies Used

* Python
* NumPy
* OpenCV
* TensorFlow / Keras
* Pickle
* Jupyter Notebook
* Kaggle Dataset

## Image Preprocessing

Before prediction, the input image is processed to match the format expected by the trained model.

The webcam application performs the following preprocessing:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

processed_frame = cv2.resize(gray, (200, 200))

processed_frame = processed_frame.astype(np.float32) / 255.0

processed_frame = processed_frame.reshape(1, 200, 200, 1)
```

Therefore, the model expects:

* Grayscale images
* Image size: **200 × 200**
* Pixel values normalized between **0 and 1**
* Input shape: **(1, 200, 200, 1)**

## Classes

The model uses the following label mapping:

```python
label = {
    'men': 0,
    'women': 1
}
```

During prediction, the numerical output is converted back into the corresponding class name.

## Model

The trained model is saved as:

```text
model.pkl
```

The model is loaded using Python's `pickle` module:

```python
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

The model receives a preprocessed image and returns a prediction for the corresponding class.

## Real-Time Webcam Prediction

The project includes `camera.py`, which allows the trained model to be used with a computer webcam.

The application:

1. Opens the default webcam.
2. Captures video frames.
3. Converts each frame to grayscale.
4. Resizes the frame to 200 × 200.
5. Normalizes the pixel values.
6. Sends the image to the trained model.
7. Displays the predicted class on the screen.

The prediction is displayed directly on the webcam window.

### Run the Webcam Application

Install the required packages:

```bash
pip install opencv-python numpy
```

Then run:

```bash
python camera.py
```

A webcam window will open.

Press:

```text
Q
```

to exit the application.

## Installation

Clone the repository:

```bash
git clone https://github.com/HovsepBadalyan/GenderClassification.git
```

Move into the project directory:

```bash
cd GenderClassification
```

Install the dependencies:

```bash
pip install numpy opencv-python tensorflow jupyter
```

## Running the Training Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Untitled.ipynb
```

Download the dataset from Kaggle and update the dataset path in the notebook if necessary.

Then run the notebook cells sequentially to:

* load the images;
* preprocess the data;
* create training and testing datasets;
* train the model;
* evaluate the model;
* save the trained model.

## Using the Trained Model

After training, the model can be saved as:

```text
model.pkl
```

The trained model can then be used independently from the training notebook.

For example:

```python
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

The model can then be used for image classification after applying the same preprocessing used during training.

## Important Considerations

This project is intended as an **educational computer vision and machine learning project**.

Predictions from an image should not be interpreted as determining a person's actual gender identity. The model only learns patterns present in the training data and may produce incorrect predictions for images that differ from the training distribution.

Model performance can be affected by:

* image quality;
* lighting;
* camera angle;
* background;
* face position;
* image resolution;
* differences between training and real-world images.

## Future Improvements

Possible improvements include:

* Use a dedicated face-detection model before classification.
* Detect multiple people in the same frame.
* Crop the detected face instead of processing the entire frame.
* Add data augmentation.
* Improve the training/validation split.
* Compare different CNN architectures.
* Use transfer learning with models such as MobileNet, ResNet, or EfficientNet.
* Add prediction confidence.
* Create a web interface for image upload.
* Deploy the model as an API.
* Improve real-time webcam performance.

## Dataset and Project Links

**Kaggle Dataset:**
https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women

**GitHub Repository:**
https://github.com/HovsepBadalyan/GenderClassification

## Author

**Hovsep Badalyan**

GitHub:
https://github.com/HovsepBadalyan

---

## License

This project is intended for educational and research purposes.

Please check the original Kaggle dataset page for the dataset's specific license and usage restrictions before redistributing the dataset.
