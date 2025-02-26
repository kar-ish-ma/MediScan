# MediScan

MediScan is an advanced health analytics platform that leverages Machine Learning (ML), Deep Learning (DL), Optical Character Recognition (OCR), and Natural Language Processing (NLP) to analyze medical data. The platform captures images from a local device's webcam, extracts text using OCR, and applies NLP techniques to mask, unmask, and predict text in the unmasked areas. Additionally, it uses Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to identify and predict handwritten text from images.

## Features

### Webcam Image Capture:
- Capture images directly from the user's local device webcam.
- Store the captured images for further processing.

### OCR for Text Extraction:
- Use Tesseract OCR to extract text from the captured images.

### NLP for Masking and Unmasking:
- Apply NLP techniques to mask sensitive information in the extracted text.
- Predict and unmask text in specific areas.

### Deep Learning for Handwritten Text Recognition:
- Use CNNs to identify and mark individual letters in handwritten text.
- Use LSTMs to predict each letter from the identified regions.
- The deep learning model is trained on the IAM Handwriting Word Database.

### User-Friendly Interface:
- Simple and intuitive interface for capturing images, viewing results, and navigating the platform.

## Technologies Used

### Machine Learning:
- OCR (Optical Character Recognition)
- NLP (Natural Language Processing)

### Deep Learning:
- CNN (Convolutional Neural Networks)
- RNN (Recurrent Neural Networks)
- CRNN (Convolutional Recurrent Neural Networks)
- LSTM (Long Short-Term Memory)

### Frameworks and Libraries:
- TensorFlow
- Keras
- PyTesseract
- OpenCV

### Backend:
- Node.js
- Express.js

### Frontend:
- HTML, CSS, JavaScript
- EJS (Embedded JavaScript templates)

### Database:
- IAM Handwriting Word Database (for training the deep learning model)

## How It Works

### Image Capture:
- The user captures an image using their local device's webcam.
- The image is stored in the uploads directory for further processing.

### Text Extraction:
- The stored image is passed to the OCR engine (Tesseract) to extract text.

### NLP Processing:
- The extracted text is processed using NLP techniques to mask sensitive information.
- The platform predicts and unmasks text in specific areas.

### Handwritten Text Recognition:
- For handwritten text, the image is processed using a CNN to identify and mark individual letters.
- An LSTM model predicts each letter from the identified regions.

### Result Display:
- The processed text (both OCR and handwritten) is displayed to the user in a clean and readable format.
  ![Alt text](https://github.com/kar-ish-ma/MediScan/blob/main/mediScanimag.jpg)


## Setup and Installation

### Prerequisites
- Python 3.x
- Node.js and npm
- Tesseract OCR installed on your system
- TensorFlow and Keras for deep learning models

### Steps to Run the Project

#### Clone the Repository:
```bash
git clone https://github.com/your-username/mediscan.git
cd mediscan
```

#### Install Python Dependencies:
```bash
pip install -r requirements.txt
```

#### Install Node.js Dependencies:
```bash
npm install
```

#### Set Up Tesseract OCR:
- Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
- Ensure the Tesseract executable is added to your system's PATH.

#### Run the Backend Server:
```bash
node server.js
```

#### Access the Application:
- Open your browser and navigate to `http://localhost:3000`.

## Project Structure
```
mediscan/
├── public/                  # Static files (CSS, JS, images)
├── views/                   # EJS templates
│   ├── home.ejs             # Home page
│   ├── upload.ejs           # Upload page
│   ├── result.ejs           # Result page
├── server.js                # Backend server (Node.js/Express)
├── test.py                  # Python script for OCR and NLP
├── requirements.txt         # Python dependencies
├── package.json             # Node.js dependencies
├── README.md                # Project documentation
```

## Training the Deep Learning Model

The deep learning model for handwritten text recognition is trained on the IAM Handwriting Word Database. To train the model:

### Download the IAM Dataset:
- Download the dataset from the IAM Handwriting Database.

### Preprocess the Data:
- Resize images to a consistent size.
- Normalize pixel values.

### Train the Model:
- Use the provided Python script (`train_model.py`) to train the CNN and LSTM models.
- Save the trained model for inference.

### Integrate the Model:
- Load the trained model in the `test.py` script for predicting handwritten text.

## Future Enhancements

### Support for Multiple Languages:
- Extend OCR and NLP capabilities to support multiple languages.

### Improved Handwritten Text Recognition:
- Train the model on larger datasets for better accuracy.

### Cloud Integration:
- Store and process images in the cloud for scalability.

### Mobile App:
- Develop a mobile app for easier access and usage.

## Contributors
- **Karishma Rahaman**

## Acknowledgments
- Tesseract OCR for text extraction.
- IAM Handwriting Database for providing the dataset for training.
- TensorFlow and Keras for deep learning model development.

