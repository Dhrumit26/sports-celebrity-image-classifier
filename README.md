# Sports Celebrity Image Classification

<img width="575" alt="Screenshot 2024-10-31 at 2 33 47 AM" src="https://github.com/user-attachments/assets/317729b2-2c32-416e-9aae-16e01772b2fc">

A Flask-based image classification application that identifies sports celebrities using Support Vector Machine (SVM) and other machine learning algorithms. This project includes data preprocessing, model training, and deployment.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Acknowledgements](#acknowledgements)
8. [License](#license)

---

### Project Overview

This application classifies images of sports celebrities using machine learning algorithms. It utilizes a custom dataset, with images preprocessed using techniques like Haar Cascade for face detection and Wavelet Transformation for feature extraction. This project demonstrates the end-to-end flow of image classification with Flask as the web server framework.

### Features

- Face detection using OpenCV and Haar Cascade.
- Feature extraction with Wavelet Transformation.
- Model training using Scikit-Learn pipelines with algorithms like SVM, Random Forest, and Logistic Regression.
- Hyperparameter tuning with Grid Search.
- Web server implementation using Flask to classify uploaded images.

### Installation

To set up and run this project, follow these steps:

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/sports-celebrity-image-classifier.git
cd sports-celebrity-image-classifier
```

#### 2. Set up a virtual environment

Using Python 3.10 (recommended for compatibility):

```bash
python3.10 -m venv myenv
source myenv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download pre-trained models and other artifacts

Place the pre-trained model (`saved_model.pkl`) and the `class_dictionary.json` file in the root directory or as per the specified path in `app.py`.

### Project Structure

```
sports-celebrity-image-classifier/
├── dataset/                   # Original and cropped images of celebrities
├── server/                    # Flask server code
│   ├── app.py                 # Main application file
│   ├── util.py                # Utility functions for model loading and processing
│   └── wavelet.py             # Wavelet Transformation for feature extraction
├── saved_model.pkl            # Trained model file
├── class_dictionary.json      # Class labels and mappings
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation
```

### Usage

#### 1. Run the Flask server

Activate your virtual environment and start the Flask app:

```bash
source myenv/bin/activate
python server/app.py
```

The server will run at `http://127.0.0.1:5000` by default.

#### 2. Access the API

Use tools like Postman or `curl` to classify an image:

```bash
curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/classify_image
```

#### 3. Response

The server will return a JSON response with the predicted class and confidence score.

### Model Training

If you wish to retrain or fine-tune the model:

1. **Prepare the Dataset**:

   - Place images in `dataset/` with subfolders for each celebrity class.
   - Use `get_cropped_image_if_2_eyes` and `w2d` in `wavelet.py` for preprocessing.

2. **Train the Model**:
   Run the model training cells in `sports_person_classifier.ipynb` to preprocess the data, train models, and save the best model.

3. **Save the Model and Class Mappings**:
   Save the model as `saved_model.pkl` and the class dictionary as `class_dictionary.json` for use in the Flask app.

### Acknowledgements

- **OpenCV** for face detection.
- **PyWavelets** for wavelet transformation.
- **Scikit-Learn** for model training and evaluation.

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
