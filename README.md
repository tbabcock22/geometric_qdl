# Quantum Machine Learning Tutorials

This repository contains two Python scripts demonstrating quantum machine learning techniques using PennyLane and Qiskit, along with an Anaconda environment file for setting up the necessary dependencies.

## Contents

- `tutorial_geometric_qml.py`: A tutorial script showcasing the use of PennyLane Geometric for graph-based machine learning tasks.
- `tutorial_qiskit.py`: A tutorial script demonstrating the use of Qiskit for quantum machine learning, specifically for training a quantum circuit to play Tic-Tac-Toe.
- `environment.yml`: An Anaconda environment file specifying the required dependencies for running the tutorial scripts.

## Prerequisites

To run the tutorial scripts, you need to have Anaconda or Miniconda installed on your system. You can download and install Anaconda from the official website: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)

## Setup

1. Clone this repository to your local machine: `git clone https://github.com/your-username/your-repository.git`

2. Navigate to the cloned repository: `cd your-repository`

3. Create the Anaconda environment using the provided `environment.yml` file: `conda env create -f environment.yml`

4. Activate the created environment: `conda activate geoQMLEnv`

## Running the Tutorials

### Tutorial: PennyLane Geometric

To run the PennyLane Geometric tutorial, execute the following command:

`python tutorial_geometric_qml.py`

This script demonstrates the usage of PennyLane Geometric for graph-based machine learning tasks.

### Tutorial: Qiskit

To run the Qiskit tutorial, execute the following command:

`python tutorial_qiskit.py`

This script showcases the use of Qiskit for quantum machine learning, specifically training a quantum circuit to play Tic-Tac-Toe. The script creates a dataset of Tic-Tac-Toe game states, builds a quantum circuit, and trains the circuit using the dataset. The script also compares the performance of a symmetric and a non-symmetric encoding of the game states.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [PennyLane Intro to Geometric QML](https://pennylane.ai/qml/demos/tutorial_geometric_qml/)
- [Qiskit](https://qiskit.org/)
