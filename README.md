# Integrated YOLOv10 and TinyLIC Compression Model

This project integrates the YOLOv10 object detection model with the TinyLIC image compression model to jointly train both models. The goal is to achieve a model that not only compresses images effectively but also allows accurate object detection on the compressed images.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training Workflow](#training-workflow)
  - [Stage 1: Pre-training Models](#stage-1-pre-training-models)
  - [Stage 2: Integrated Model Training](#stage-2-integrated-model-training)
- [Parameters](#parameters)
- [License](#license)

## Installation

### Prerequisites

Ensure that you have the following installed:
- Python 3.6 or higher
- PyTorch
- ultralytics package (for YOLOv10)
- compressai package (for TinyLIC)
- CUDA (optional, for GPU support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fumchin/Video-Compression-for-Object-Detection.git
   cd vcod
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
|-- compressai/          # Compression model directory
|-- ultralytics/         # YOLOv10 model directory
|-- checkpoints/         # Pretrained weights directory
|-- scripts/             # Training and utility scripts
|-- README.md            # Project documentation
|-- VOC.yaml             # Dataset configuration
|-- merge.py             # Main script to train integrated model
```

## Usage

### Training Workflow

The training process is divided into two main stages:

#### Stage 1: Pre-training Models

- **YOLOv10**: Fine-tune the YOLOv10 model using the VOC dataset.
- **TinyLIC**: Use pre-trained TinyLIC models directly.

#### Stage 2: Integrated Model Training

In this stage, the pre-trained YOLOv10 and TinyLIC models are combined, and the integrated model is trained to optimize both object detection and image compression simultaneously.

### Running the Training Script

1. Pre-train the YOLOv10 and TinyLIC models separately if not done already.
   
2. Start the integrated training by running the `merge.py` script:

   ```bash
   python merge.py
   ```

   This script will:
   - Load pre-trained YOLOv10 and TinyLIC models.
   - Train the integrated model using the VOC dataset.

3. The training configuration, such as the number of epochs, batch size, and image size, can be adjusted directly in the `merge.py` script.

## Parameters

- `Lambda`: Enum for different compression levels (`q1`, `q3`, `q6`, `q8`).
- `lr`: Learning rate for optimizers.
- `imgsz`: Image size for training (default is 256).
- `batch`: Batch size for training (default is 4).
- `epochs`: Number of epochs for training (default is 400).
- `data`: Path to the dataset configuration file.
- `base_dir`: Directory to save model checkpoints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
