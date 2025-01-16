# Dog Breed Identifier Model [API]

## Overview

The Dog Breed Identifier is a Python-based application that utilizes a pre-trained ResNet50V2 model to classify images of dogs into their respective breeds. The model is fine-tuned and trained on a dataset of dog images labeled with their breeds, enabling accurate breed identification. I used it as an API to WhoofMyDog - university lab project.

## Features

- **Data Preparation**:
  - Reads breed labels from `labels.csv`.
  - Processes images from the `train` folder, resizing and preprocessing them for model training.
  - Splits the dataset into training and testing sets to evaluate model performance.

- **Data Augmentation**:
  - Applies image data augmentation techniques to the training set, enhancing the model's ability to generalize to new, unseen images.

- **Model Training**:
  - Employs a ResNet50V2 model pre-trained on ImageNet.
  - Customizes the top layers of the model for dog breed classification.
  - Trains the model over multiple epochs with callbacks for learning rate reduction and early stopping to prevent overfitting.

- **Prediction**:
  - Classifies input images of dogs, predicting the breed with high accuracy.

## Installation

To run the Dog Breed Identifier locally, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.2 or higher
- OpenCV
- NumPy
- Pandas
- Matplotlib

You can install the required packages using pip:

```bash
pip install requirements.txt
```

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/zecozejak/Dog-Breed-Identifier.git
   cd Dog-Breed-Identifier
   ```

2. **Prepare the dataset**:

   - Ensure that the `train` folder contains the training images of dog breeds.
   - Verify that `labels.csv` accurately maps image IDs to breed names.

3. **Run the application**:

   Execute the main Python script to train the model and make predictions:

   ```bash
   python dog-breed-identification.py
   ```

   This script will train the model and predict the breed of a specified dog image. You can modify the image path in the code to test different images. The output will display the predicted breed name for the input image.

## Project Structure

- `dog-breed-identification.py`: Main script containing code for data preparation, model training, and prediction.
- `labels.csv`: CSV file mapping image IDs to breed names.
- `model/`: Directory containing the saved model and weights.
- `test/`: Folder with test images of various dog breeds.
- `train/`: Folder containing training images for 60 dog breeds.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the contributors and the open-source community for their invaluable resources and support. 
