# TOKAI POC

## Overview
Quick demo tokai tagname/barcode detection using openvivo library, support intel device.

## Installation
### Requirements
- Python 3.9

### Setup
1. Create a Conda environment:
    ```bash
    conda create --name tokai_poc python=3.9
    ```

2. Activate the created environment:
    ```bash
    conda activate tokai_poc
    ```

3. Install required packages using pip:
    ```bash
    pip install -q "openvino-dev>=2023.0.0" "nncf>=2.5.0"
    pip install -q "ultralytics==8.0.206" onnx
    ```

## Usage
To test the environment and run the YOLOv8 script, execute the following command in the activated Conda environment:

```bash
python openvivo_yolov8.py

