# Deep Learning Anomaly Detection in Time Series

## Project Overview

This repository demonstrates advanced anomaly detection techniques in time series data. Anomaly detection is a critical application in numerous domains, including IT infrastructure monitoring, fraud detection, predictive maintenance, and cybersecurity. The project specifically focuses on identifying irregular spikes in computational resource usage, utilizing real-world EC2 CPU utilization data.

Three primary methodologies have been implemented and rigorously assessed:
1.  **Median Absolute Deviation (MAD)**: A robust, statistical baseline model.
2.  **Isolation Forest**: An unsupervised machine learning approach employing ensemble methods.
3.  **Local Outlier Factor (LOF)**: A density-based machine learning algorithm for anomaly detection.

This project sets the foundation for expanding into deep learning architectures, such as Long Short-Term Memory (LSTM) networks, for temporal sequence anomaly detection.

## Repository Structure

The project has been organized to adhere to standard software engineering practices:

```text
Deep-Learning-Anomaly-Detection-TimeSeries-LSTM/
├── data/
│   ├── ec2_cpu_utilization.csv      # Time series data of CPU usage
│   └── combined_labels.json         # Ground truth anomaly labels
├── Anomaly_Detection_TimeSeries_LSTM.ipynb # The core analysis notebook
└── README.md                        # Project documentation
```

## Dataset Description

The dataset employed is part of the Numenta Anomaly Benchmark (NAB) and comprises actual CPU utilization metrics from an Amazon Web Services (AWS) EC2 instance over an extended period. 

*   **Data Interval**: Readings were collected every 5 minutes.
*   **Total Data Points**: 4,032 sequential observations.
*   **Features**: `timestamp` (the exact time of the reading) and `value` (the actual CPU utilization metric).
*   **Labels**: Anomalies corresponding closely to known performance degradation events are defined separately in the JSON file.

## Methodologies Detailed

### 1. Median Absolute Deviation (MAD)

As a fundamental step in anomaly detection, a robust statistical baseline must first be established. The Median Absolute Deviation is preferred over standard deviation because it is highly resistant to extreme outliers in the dataset.

*   **Mechanism**: A robust Z-Score is computed for every data point using the formula: `Z = 0.6745 * (x - median) / MAD`.
*   **Thresholding**: Any observation that yields a robust Z-Score exceeding an absolute value of 3.5 is categorized as an anomaly.
*   **Performance**: MAD is exceptionally fast and computationally inexpensive, making it highly effective for catching obvious spikes without being negatively skewed by those spikes.

### 2. Isolation Forest

To introduce a machine learning approach, the Isolation Forest algorithm is utilized. It is explicitly designed for anomaly detection.

*   **Mechanism**: The algorithm constructs multiple random decision trees. Because anomalies are "few and different," they are isolated closer to the root of the trees compared to normal observations.
*   **Isolation Path**: The shorter the average path length required to isolate a data point, the higher the likelihood that the point is an anomaly.
*   **Performance**: Highly adaptable and suited for high-dimensional scenarios, although it introduces hyperparameters such as the `contamination` rate (the estimated percentage of outliers in the data).

### 3. Local Outlier Factor (LOF)

Another unsupervised approach explored in this project is the Local Outlier Factor.

*   **Mechanism**: LOF computes the local density deviation of a given data point with respect to its neighbors. It flags observations that have a substantially lower density than their neighbors.
*   **Local Density**: It effectively captures anomalous points in datasets where anomalies might not be global extreme values but rather local irregularities.
*   **Performance**: Excellent at finding local outliers, but can be slightly more computationally intensive and requires careful neighbor parameter configurations.

## Future Directions: Deep Learning

While statistical methods and isolation forests are effective, they often analyze each point independently and fail to fully comprehend the underlying sequential structure of time series data. Therefore, the future direction of this project is to integrate **Long Short-Term Memory (LSTM)** neural networks. 

LSTM models excel at learning long-term dependencies in sequential data. An LSTM-based autoencoder can learn the "normal" behavior pattern of the CPU utilization over time and flag an anomaly when the reconstruction error for a specific sequence exceeds a defined threshold.

## Usage Instructions

To execute the notebook and reproduce the findings:

1.  **Environment Setup**: Ensure that Python 3.8+ is installed.
2.  **Dependencies**: Install the required scientific computing libraries.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
    ```
3.  **Execution**: Launch the Jupyter Notebook environment localized to the project directory.
    ```bash
    jupyter notebook Anomaly_Detection_TimeSeries_LSTM.ipynb
    ```
4.  **Analysis**: The notebook has been structured meticulously with detailed markdown annotations at every step to facilitate comprehension without the requirement of running the cells immediately.

## Conclusion

The exploration of Median Absolute Deviation, Isolation Forests, and the Local Outlier Factor provides substantial insight into establishing baseline anomaly detection solutions. The findings substantiate the need for careful threshold tuning and highlight the value of objective ground truth labels. Implementing these methods is paramount before scaling up to more computationally intensive deep learning architectures.
