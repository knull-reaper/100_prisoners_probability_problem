# 100 Prisoners Problem Simulation

This repository contains a Python script for simulating the 100 prisoners problem using CUDA for high-performance computation. The script can be used to verify the ~31% success rate of the optimal strategy.

## Features

- **High-Performance:** Utilizes NVIDIA CUDA for massively parallel simulations.
- **Interactive Mode:** Allows you to choose the number of prisoners and trials.
- **Batch Processing:** Automatically handles large-scale simulations that don't fit into GPU memory at once.
- **Local & Global Memory Kernels:** Switches between different CUDA memory strategies based on the problem size for optimal performance.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the script:**
    ```bash
    python 100_prisoners_probability.py
    ```
3.  Follow the on-screen prompts to select the number of prisoners and trials.

## Simulation Results

The simulation confirms the theoretical predictions and demonstrates the practical application of the strategy across different scales. The success rate consistently hovers around 31%, even with a large number of prisoners.

| Prisoners (N) | Trials    | Success Rate | Kernel | Batch Mode |
| :------------ | :-------- | :----------- | :----- | :--------- |
| 100           | 1,000,000 | 31.2919%     | Local  | No         |
| 1,000         | 100,000   | 30.8900%     | Local  | No         |
| 10,000        | 10,000    | 30.2100%     | Global | No         |
| 100,000       | 10,000    | 30.2100%     | Global | Yes        |
