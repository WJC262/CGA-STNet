# CGA-STNet: Dockless Shared Bicycle Demand Prediction

This repository contains the implementation of CGA-STNet, a model designed for predicting dockless shared bicycle demand, incorporating multiple spatial features and time periodicity.

## Directory Structure

- **dataloader构建**: Contains a CSV file with research information.
  - `研究范围信息.csv`

- **layers**: Contains the model layers and implementations.
  - `__init__.py`
  - `Conv_Blocks.py`
  - `Embed.py`

- **output**: Directory for output files.
  - `CNN_TimesNet_cga.py`

- **main.py**: Main script to run the model.

## Requirements

- Python 3.8+
- Required libraries:
  - warnings
  - pandas
  - numpy
  - time
  - torch
  - tqdm
  - csv
  - datetime
  - pickle
  - gc


## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>

## Contact

For any inquiries or issues, please contact the authors at:

- qianhq@emails.bjut.edu.cn
- jiachen_wang1@brown.edu
- cdyan@bjut.edu.cn
