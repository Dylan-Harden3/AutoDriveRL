# CSCE 642 Project

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway-env.gif?raw=true"><br/>
</p>

## Setup
You can set up the environment using conda + pip or virtual env + pip, Python version 3.10.12 is required.

To install the packages run
```bash
pip install -r requirements.txt
```

## Running the Agents
To run the agents you simply run
```bash
python3 main.py <args>
```
A full list of arguments can be found at the bottom, to produce our results we ran the following commands.
#### DQN
For Training:
For DQN we used these hyperparameters with alpha set to 0.0, 0.5 and 1.0.
```bash
python3 main.py -s 5000 -n 1024 -l 0.0001 -g 0.99 -d 90 -E 0.9 -m 1000 -N 100 -B 32 -a <alpha> -L 5 -S ddqn -M train
```
And for Testing:
```bash
python3 main.py -S ddqn -M test -e 100 -t 1000
```

#### A2C
For Training:
```bash
python3 main.py -s 20000 -n 2048 -l 0.0001 -g 0.99 -d 90 -S a2c -M train
```
And for Testing:
```bash
python3 main.py -S a2c -M test -e 100 -t 1000
```

***Note that for -M test we expect there to already be trained models for all 3 alpha values (0.0, 0.5, 1.0). Our trained models are already saved.***
## Plotting Data
Once the data has been collected:
```bash
python3 main.py -S <any_solver> -M plot_both
```
***Note that plotting will only occur if the corresponding pickle files are present and in the correct directory.***
## Command-line Arguments
You must specify the `-S` and `-M` flags to specify training and which solver or testing.
All other arguments are optional and have default values configured as described below.
To list out all the arguments you can use the `-h` flag.
| Argument | Description                                                                                    | Default Value |
|----------|------------------------------------------------------------------------------------------------|---------------|
| `-e`     | Number of episodes for evaluation (only needed with `-M test`)                                 | 100           |
| `-s`     | Number of environment steps to train for                                                       | 5000          |
| `-t`     | Max steps per episode in testing (only needed with `-M test`)                                  | 1000          |
| `-n`     | Number of dense neurons per layer                                                              | 1024          |
| `-l`     | Learning rate                                                                                  | 0.0001        |
| `-g`     | Discount factor                                                                                | 0.99          |
| `-d`     | Max duration per episode (seconds)                                                             | 90            |
| `-E`     | Initial Epsilon for Epsilon-Greedy in DQN (We use linear decay to 0 over the first 5000 steps) | 0.9           |
| `-m`     | Max Replay Memory size for DQN                                                                 | 1000          |
| `-N`     | Interval for updating target newtork in DQN (number of steps)                                  | 100           |
| `-B`     | Batch Size for sampling for DQN Replay Memory                                                  | 32            |
| `-a`     | Alpha for Prioritized Experience Replay                                                        | 0.0           |
| `-L`     | Number of Dense Layers for DQN Q network                                                       | 5             |
| `-S`     | Which solver to run for training, one of ddqn or a2c                                           | None          |
| `-M`     | Mode, either train, test, or plot_both                                                         | None          |

## Citation
The source code for the environment requests that we include this citation.
```bash
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```
