# LIVCell Segmentation Project

This project performs semantic segmentation on medical cell images ([LIVECell dataset](https://github.com/sartorius-research/LIVECell)) using a U-Net architecture. The repository includes the necessary files for training, testing, data handling, deployment via Docker, and inference.

## Repository Structure

Ensure the following are installed on your machine:

* Python 3.8+
* Docker Desktop
* Jupyter Notebook (optional, for training)

## Files Overview

1. `data_torch.py`
This file contains the `LIVCellDataset` class, a PyTorch-compatible dataset loader:

* Reads images and annotations from a directory.
* Applies transformations for data augmentation and normalization.

2. `model_training.ipynb` This Jupyter notebook contains the code for:
* Loading the data using a `LIVCellDataset` class.
* Training the U-Net model on the dataset.
* Testing and evaluating the model's performance.


Usage:

* Open the notebook.
* Configure the paths for the dataset and annotations.
* Run the cells to train and test the model.

3. `Dockerfile`
The Dockerfile sets up an environment to deploy the trained model for inference using FastAPI:

* Starts with a base Python image.
* Installs dependencies from requirements.txt.
* Copies the FastAPI application (main.py) and model file (unet_model_48.pth) into the container.
* To build the Docker image:

`docker build -t unet-segmentation-api .`

4. `main.py`
A FastAPI-based script to handle image uploads and return predicted segmentation masks:

* Accepts an image file via HTTP POST.
* Runs inference using the trained U-Net model.
* Returns the segmentation mask as an image file.

To run the docker

`curl -X POST -F "file=@path_to_image/image.tif" http://localhost:8000/predict -o output_mask.png`

5. `unet_model_28.pth`

The trained U-Net model weights file, saved in PyTorch's .pth format. It is loaded during deployment for inference.

7. `requirements.txt`

Lists all required Python libraries, including:
```
flask
fastapi
uvicorn
torch
torchvision
pillow
numpy
matplotlib
opencv-python
python-multipart
```

Install dependencies with:

`pip install -r requirements.txt`

## How to Run

1. Training the Model
Open `model_training.ipynb`.

Configure the dataset paths and model parameters. Run the notebook to train the model and save the weights.

3. Testing the Model
Use the testing code in the notebook to evaluate model performance.

5. Running the Docker Container
Build the Docker image:

`docker build -t unet-segmentation-api .`

Run the Docker container:

`docker run -p 8000:8000 unet-segmentation-api`

7. Making Predictions
Send a POST request to the /predict endpoint with an image file:

`curl -X POST -F "file=@path_to_image/image.tif" http://localhost:8000/predict -o output_mask.png`

The predicted segmentation mask will be saved as output_mask.png.

## TO-DO

* Calculate Precession and Recall values, along with Jaccard Index along with model AP
* Test other models for accuracy and latency
* Evaluate quantitaviely as well as visually the examples with bad accuracy metric and try and understand the edge cases

## References

* [U-Net PyTorch Example](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)
