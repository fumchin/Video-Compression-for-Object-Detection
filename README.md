# VCOD Project

VCOD is an efficient image compression and object detection project aimed at providing advanced compression algorithms and accurate object detection models.

## Directory Structure


## File Descriptions

### Main Files

- `app.py`: The main application file responsible for starting and running the entire project.
- `continue_train.py`: Script for continuing model training.
- `eval.py`: Script for evaluating model performance.
- `evaluate_psnr.sh`: Script for evaluating PSNR (Peak Signal-to-Noise Ratio).
- `coco.yaml`: Configuration file for the COCO dataset.
- `merge.py`: Script for merging multiple models or datasets.

### Functionality of `merge.py`

`merge.py` is a crucial script responsible for merging multiple models or datasets into a unified model or dataset. Its main functionalities include:

1. **Reading Input**: Reads multiple models or datasets from specified directories or files.
2. **Data Processing**: Preprocesses the read data, including data cleaning and format conversion.
3. **Merging Data**: Merges the weights of multiple datasets or models into a unified dataset or model.
4. **Saving Results**: Saves the merged dataset or model to a specified directory or file.

Below is a simplified example code of `merge.py`:

```python
import os
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def merge_data(data_list):
    merged_data = {}
    for data in data_list:
        for key, value in data.items():
            if key in merged_data:
                merged_data[key].extend(value)
            else:
                merged_data[key] = value
    return merged_data

def save_data(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file)

def main(input_paths, output_path):
    data_list = [load_data(path) for path in input_paths]
    merged_data = merge_data(data_list)
    save_data(merged_data, output_path)

if __name__ == "__main__":
    input_paths = ["data1.json", "data2.json"]
    output_path = "merged_data.json"
    main(input_paths, output_path)
