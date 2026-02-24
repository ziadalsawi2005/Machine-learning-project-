# Network Intrusion Detection System (NIDS)

A production-ready Machine Learning-based Network Intrusion Detection System designed to detect various types of network attacks including DDoS, Port Scanning, Web Attacks, and Infiltration attempts.

## Features

- Binary and multi-class classification capabilities
- Automated data preprocessing pipeline
- Multiple ML algorithms: Logistic Regression, Decision Tree, Random Forest
- Comprehensive model evaluation metrics
- Automatic best model selection based on recall score
- Production-ready code structure

## Project Structure

```
project/
├── data/                    # Dataset files
├── models/                  # Trained model artifacts
├── src/
│   ├── preprocessing.py     # Data preprocessing utilities
│   ├── train.py            # Model training logic
│   ├── evaluate.py         # Model evaluation functions
│   ├── predict.py          # Prediction module
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your dataset files in the `data/` directory

## Usage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% very important :
please to run this program please try to dowenload the data to train the model i sugguest for you " https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download " please dowenload it and but it in folder named it " data " ok 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Run the complete pipeline:
```
python main.py
```

Or run individual components:
```
python src/train.py
python src/predict.py --input_file=path/to/test_data.csv
```

## Security Considerations


This system prioritizes minimizing false negatives (attacks that go undetected) which are critical in security contexts. The model selection process specifically emphasizes recall as the primary metric since missing an attack is far worse than flagging normal traffic as suspicious.

