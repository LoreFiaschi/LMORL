# LMORL suite: a Lexicographic Multi-Objective Reinforcement Learning suite



## Characteristics

- Expose agents developed in multiple languages (**Python** and **Julia**).
- Has a simple standard interface that is compatible with [Gymnasium's environment API](https://gymnasium.farama.org/api/env/).
- Exploits **Non Archimedean** scalarization in order to solve Multi-Objective RL problems through the usage of **BANs** (Bounded Algorithmic Numbers).
- Easily allow to add other agents by implementing the standard agent class.
- Gives a suite of tools to visualize and analyze the results of learning tasks.
- Allow to store and load trained agents to file.

## Installation and first setup

This library requires multiple dependencies both from other Python packages and both
from Julia packages.

### Julia required environment and packages
Since some Python packages will need a working Julia environment in order to be correctly
setup, you have to install a Julia interpreter globally. The suggested version is 1.9.
[Julia download page](https://julialang.org/downloads/)

### Julia dependencies installation
To install the required dependencies for Julia, you can run the script julia-dependencies.jl
located in the `/setup/` folder.
```
julia julia-dependencies.jl 
```

### Python dependencies
In `LMORL/BAN/requirements.txt` there are the Python packages dependencies, that can
be installed by running
 ```
 pip install -r requirements.txt
 ```

### Julia Python interface first setup
To install the required dependencies for Julia to be run from Python, you can run the
script `julia-api-installation.py` located in the `/setup/` folder.
```
python julia-api-installation.py
```
### Mo Gymnasium Box2D dependencies
In order to use the Lunar Lander Gymnasium environment, you have to run 
```
pip install gymnasium[Box2d]
```
which on some configurations can lead to missing Visual C++ dependencies error.
You can find the right version of Visual C++ for your system [here](visualstudio.microsoft.com/visual-cpp-build-tools).

## Check if the library is properly working

Once installed all the dependencies you can check if everything was configured fine by firstly importing a **DQNHybrid** agent and then by instantiating it, which needs of all the components of the library to work properly.

```
import sys
import os
from pathlib import Path

root_dir = "PATH/TO/LMORL/REPO/ROOT/FOLDER"

if root_dir not in sys.path:
  sys.path.append(root_dir)

from LMORL.BAN.API.agents.DQNHybrid import DQNHybrid

```

In order to check if also the Gymnasium library and the *Lunar Lander* environment were configured correctly, you can try to load a MO LL environment by running the following code:

```
import mo_gymnasium as mo_gym

env = mo_gym.make("mo-lunar-lander-v2", render_mode="rgb_array")
```

Once you have both an environment and the **DQNHybrid** class imported, you can instantiate the agent:

```
input_size = env.observation_space.shape[0]
num_actions = int(env.action_space.n)
action_space = list(range(env.action_space.n))
learning_rate = 0.001
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
train_start = 1000
hidden_size = 128
BAN_SIZE = 3
max_memory_size=100000

agent = DQNHybrid(input_size=input_size, num_actions=num_actions,
                  action_space=action_space, learning_rate=learning_rate,
                  epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                  batch_size=batch_size, hidden_size=hidden_size,
                  ban_size=BAN_SIZE, max_memory_size=max_memory_size, train_start=train_start)
```


## Usage example

You can find various usage examples in the [examples](/examples/) folder.