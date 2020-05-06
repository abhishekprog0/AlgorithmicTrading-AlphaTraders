# AlgorithmicTrading-AlphaTraders

The directory structure of the project can be seen below.

```bash
.
├── data
│   ├── combined
│   └── trade_data
├── saved_models
├── source
│   ├── bench_mark
│   │   └── __pycache__
│   ├── execution_source
│   │   └── __pycache__
│   └── __pycache__
├── Stock Selection Sector Files
└── test_data

```


We will describe the directories below. 

* <strong> Data </strong> : The data directory contains all the data that we have used to train and test the models. Within the data directory, there are two subdirectories called combined and trade_data. The combined directory contains CSV files that contain the combined features for the stocks that we have used to train and test the data. The trade directory contains the tick level price and quotes data.

* <Strong> Saved models </strong> This directory state dictionaries of the saved model. These state dictionaries are used to load the model saved model when it is being tested

* <Strong> Source </strong> The source directory contains all the source code for the project. Within this directory, there are subdirectories named benchmark and execution source. The benchmark directory contains the implementations of the benchmarks that we used to compare the performance of our model. The code for execution of the trades using VWAP is present in the execution_trade directory. The execution code is largely based on the code that was provided as part of the lecture. 


## Installing the required libraries

We have created a <strong> requirements.txt </strong> file that contains all the libraries that are required to run the project. The libraries can be installed by typing the following command in the terminal.

```bash
pip install -r requirements.txt
```

## Running the project

The main files that drive the execution can of the project are called <strong> policy_train.py </strong> and can be found under the source directory. There are two different modes to run the project train and test. Before the policy can be trained or tested you would need to start the Visdom service to visualize the returns on the stock. This can be done by typing 
```bash
python -m visdom.server
```
This command will start the visdom GUI that will be used by the driver program to visualize the returns of the stock. 

</br> 
If the user wants to train a new policy it can be done by typing the following command on the terminal.

```bash
python policy_train.py --mode=test --file=combined.csv
```

After running this command the source code will start training a policy to run. To test the model and execute the trade the project needs to be run on test mode. This can be done by typing the following command. 

```bash
python policy_train.py --mode=test --file=back_test_2.csv
```

This will prompt the code to output the number of stocks that have to be traded and also invoke the VWAP trading algorithm. 