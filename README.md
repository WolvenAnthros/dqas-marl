# DQAS-MARL

Code supporting the paper Distributed quantum architecture search using multi-agent reinforcement learning

## Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Acknowledgements](#-acknowledgements)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/WolvenAnthros/dqas-marl.git

# Navigate into the project directory
cd dqas-marl

# Install requirements
pip install -r requirements.txt
```

> **Requirements**: Python 3.10

---

## Usage

### Example Script
Example MaxCut training script
```bash
python main.py --num_qubits 6 --num_agents 6 --num_train_graphs 1 --num_test_graphs 1 --max_ep_len 5

```
Example Schwinger VQE training script
```bash
python main.py --schwinger --num_qubits 4 --num_agents 4 --m 1 --max_ep_len 4  --independent
```
All passable parameters are listed in main.py with the corresponding description
---

## Acknowledgements

This code includes base implementation of code for QMIX originally written by @Lizhi-sjtu.
Original source: https://github.com/Lizhi-sjtu/DRL-code-pytorch.git

-
