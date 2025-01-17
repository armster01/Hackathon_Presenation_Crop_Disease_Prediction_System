# Crop Disease Prediction System

An AI-driven system for predicting crop diseases using Convolutional Neural Networks (CNN) and Deep Learning methods.

## Features

- Disease prediction for multiple crops (currently Apple and Corn)
- Real-time image processing and prediction
- Detailed disease information and treatment recommendations
- User-friendly web interface
- High-accuracy CNN model

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Create a `dataset` folder
   - Organize images in the following structure:
     ```
     dataset/
     └── train/
         ├── Apple___Apple_scab/
         ├── Apple___Black_rot/
         ├── Apple___Cedar_apple_rust/
         ├── Apple___healthy/
         ├── Corn___Cercospora_leaf_spot/
         ├── Corn___Common_rust/
         ├── Corn___Northern_Leaf_Blight/
         └── Corn___healthy/
     ```

4. Train the model:
   ```bash
   python train.py
   ```

5. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Upload an image of a crop leaf
3. Click "Predict Disease"
4. View the prediction results and recommendations

## Model Architecture

The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for classification
- Softmax output layer for multi-class prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
