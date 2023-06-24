
# Bus Routes Genetic Algorithm

This repository contains code for a Bus Routes Genetic Algorithm, aimed at optimizing bus routes in the bus system at Universiti Putra Malaysia (UPM). The goal of this project is to improve the consistency of bus frequencies and provide an efficient transportation system for students within the university campus.

## Code Files

The following Python files are included in this repository:

- `BusRouteOptimiser.py`: Contains the main code for the GA implementation, including the initialization, evaluation, modification, optimization, and result analysis steps.
- `MainGui.py`: Implements the graphical user interface (GUI) for the bus route optimization process and also the main method to run the program.
- `ReadData.py`: Handles data processing tasks, including reading input data and generating required matrices.

## Dataset

The `bus_dataset` folder contains the necessary input data files for the optimization process. Make sure to have the following files in the appropriate format:

- `FleetData.csv`: Contains information about the fleet size and maximum route length.
- `StopData.csv`: Provides data on bus stops, including their activity levels.
- `DemandMatrix.csv`: Represents the transportation demand matrix between bus stops.
- `LinkMatrix.csv`: Represents the travel time matrix between bus stops.

## Usage

To run the code and optimize bus routes at UPM, follow these steps:

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/sy4kila/Bus-Routes-Genetic-Algorithm.git
   ```

2. Ensure you have Python 3 installed on your machine.

3. Navigate to the cloned directory:

   ```
   cd Bus-Routes-Genetic-Algorithm
   ```

4. Make sure the required input data files are present in the `bus_dataset` folder.

5. Run the `BusRouteOptimiser.py` file to start the bus route optimization process.

6. The `MainGui.py` file can be executed to utilize the graphical user interface for interacting with the optimization process.

Please note that the specific implementation details of the code files mentioned above are not provided in the repository. You may need to refer to the code files and their internal structure for a deeper understanding of the implementation.

