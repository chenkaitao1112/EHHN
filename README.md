 #  Official Implementation

 # The main text code will be uploaded later

> **Note:**
> This repository contains the official code implementation for the paper **"EHHN: An Event-driven Heterogeneous Hypergraph Network for Object-Centric Next Activity Prediction"** .

This project proposes an event-driven Heterogeneous Hypergraph Learning approach specifically designed for Object-Centric Event Logs (OCEL) to tackle the Next-Activity Prediction task in process mining.

---

## 📁 Repository Structure

The core code and datasets are organized as follows:

- `data/`: Directory for storing raw datasets
- `newTry/`: Core implementation code
  - **Data Analysis & Preprocessing**
    - `ana.py`: Script for exploring/analyzing raw datasets
    - `construct_PE.py`: Helper functions for feature construction
    - `preprocess.py`: Helper functions for general preprocessing
    - `pipeline_OTC.py`: Dedicated pipeline for OTC dataset
    - `pipline_2017.py`: Dedicated pipeline for BPI 2017 dataset
    - `pipline_inter.py`: Dedicated pipeline for Inter dataset
    - `pipline_p2p.py`: Dedicated pipeline for P2P dataset
  - **Model Architecture**
    - `OCELhg.py`: Custom core structure (OCEL Heterogeneous Hypergraph)
    - `encoder.py`: Encoder module
    - `model.py`: Overall model architecture
  - **Training & Evaluation**
    - `Trainer.py`: Model training script
    - `test.py`: Model testing and evaluation script
  - **Configuration & Utilities**
    - `config.py`: Configuration file
    - `utils.py`: General utility functions

---

## 📖 Quick Start & Usage Guide

### 1. Environment Setup

We recommend using Conda to create a virtual environment:

```bash
conda create -n ocel_hg python=3.8
conda activate ocel_hg
```

Install required packages:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install torch_geometric pandas numpy pm4py
```

### 2. Data Preparation

Place your raw OCEL datasets into the `data/` directory. Ensure the filenames match the paths specified in your preprocessing pipelines.

### 3. Configuration

Before running any scripts, open `newTry/config.py` to configure your settings (Target Dataset, Hyperparameters, Paths, etc.).

### 4. Data Preprocessing (Crucial First Step)

We provide customized preprocessing pipelines for each dataset. Run the specific pipeline tailored to your dataset to clean data and construct the hypergraph.

For the OTC dataset:
```bash
python newTry/pipeline_OTC.py
```

For other datasets:
```bash
python newTry/pipline_2017.py
python newTry/pipline_inter.py
python newTry/pipline_p2p.py
```
*(Tip: If you are introducing a new dataset, run `newTry/ana.py` first to analyze the raw data structure.)*

### 5. Training

Once the preprocessing is complete, you can start training the model:

```bash
python newTry/Trainer.py
```

### 6. Testing & Evaluation

After the model finishes training, run the test script to evaluate its performance:

```bash
python newTry/test.py
```

---

## 🔑 Key Components
* **`OCELhg.py`**: Maps object-centric event logs into a heterogeneous hypergraph structure to capture complex many-to-many relationships.
* **Customized Pipelines**: Emphasizes refined feature engineering tailored to different business contexts.

* **`OCELhg.py`**: Maps object-centric event logs into a heterogeneous hypergraph structure to capture complex many-to-many relationships.
* **Customized Pipelines**: Emphasizes refined feature engineering tailored to different business contexts.
