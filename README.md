# Low-Level Parallel Programming - Uppsala University (Spring 2024)

This repository contains the solutions for the assignments of the Low-Level Parallel Programming course at Uppsala University, Spring 2024. The course focused on parallel programming techniques using C++ Threads, OpenMP, SIMD, and CUDA. The project involved a series of assignments that built upon each other to create a simulation of pedestrian movement using different parallelization strategies.

## Project Overview

### Assignment 1: Serial and Parallel Implementations using OpenMP and C++ Threads
- **Objective:** Implement and parallelize the `tick` function in `ped_model.cpp` to simulate the movement of agents in a 2D space.
- **Tasks:**
  - Implement the serial version of the `tick` function.
  - Identify sources of parallelism and choose suitable parallelization strategies using OpenMP and C++ Threads.
  - Implement both OpenMP and C++ Thread versions and make them interchangeable.
  - Evaluate performance by varying the number of threads and generating a speedup plot.

### Assignment 2: SIMD and CUDA Implementations
- **Objective:** Recognize vectorization possibilities and implement SIMD (Single Instruction, Multiple Data) operations. Optionally, parallelize the code using CUDA.
- **Tasks:**
  - Parallelize the code using SIMD to handle multiple agents simultaneously.
  - (Optional) Implement the same code section using CUDA for GPU acceleration.
  - Evaluate performance improvements and compare them with the serial, OpenMP, and C++ Thread versions.

### Assignment 3: Collision Prevention and Load Balancing
- **Objective:** Implement collision prevention logic and parallelize it using threads, OpenMP, or CUDA.
- **Tasks:**
  - Modify the code to use the `move` function for setting the correct positions of agents.
  - Parallelize the collision prevention logic while ensuring race-free inter-thread interaction.
  - Evaluate and compare the performance of different implementations.

### Assignment 4: Heatmap Visualization and Heterogeneous Computation
- **Objective:** Implement a visual heatmap to show the contention of locations in the simulation and parallelize its creation using CUDA.
- **Tasks:**
  - Activate the heatmap visualization in `MainWindow.cpp`.
  - Parallelize the heatmap creation steps using CUDA, ensuring no data races.
  - Implement heterogeneous computation by running collision handling on the CPU and heatmap calculation on the GPU concurrently.
  - (Bonus) Move the calculation of the next desired position to the GPU and ensure efficient data transfer between the CPU and GPU.

## Project Structure
- **libpedsim:** Contains the core simulation logic for pedestrian movement.
- **demo:** A visual representation of the simulation, using `libpedsim` for the underlying logic.
- **Assignments:** The implementations of the various assignments, including the serial, OpenMP, C++ Threads, SIMD, and CUDA versions.
- **Scenarios:** Predefined scenario files used to test and evaluate the performance of different implementations.

## Getting Started

### Prerequisites
- **C++11 or later**
- **OpenMP**
- **CUDA Toolkit** (for CUDA implementations)
- **Intel Intrinsics Guide** (for SIMD operations)

### Building the Project
1. Clone the repository:
```bash
   git clone https://github.com/odulsuzkisafilm/low-level-parallel-programming.git
   cd low-level-parallel-programming
```
2. Build the project:
```bash
   make
```

3. Run the simulation:
```bash
   ./demo/demo scenario.xml
```

!!! Check the Assignment reports for specific build guides for the project on different assignments.

### Choosing Different Implementations
You can choose between the serial, OpenMP, and C++ Thread implementations using command-line arguments or by changing a variable in the source code and recompiling.

### Evaluating Performance
You can use the -timing-mode option to run the simulation without the GUI and retrieve performance statistics.

## Results
The project reports includes generated plots showing the speedup and performance improvements achieved using different parallelization techniques. Moreover, a detailed review of the developments done in each assignment can be found in the corresponding reports.

## Acknowledgments
This project includes software from the [PEDSIM simulator](http://pedsim.silmaril.org/), a microscopic pedestrian crowd simulation system, adapted for the Low-Level Parallel Programming course at Uppsala University.

