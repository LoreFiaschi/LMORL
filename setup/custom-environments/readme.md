### How to load a custom environment in gym

#### How to find gym library directory
If your gym installation is global, you will find gym envs folder in 
Python's site-packages folder.

You can print a package folder by using this Python code (in the example it prints the path for gym):
```
import os
import gym

print(os.path.abspath(gym.__file__))
```

#### steps

* In order to create a custom environment you need to put .py environment file inside `PATH_TO_PYTHON_GYM_LIBRARY/gym/envs/box2d` for the case of lunar landers.

* Then edit the `__init__.py` inside `PATH_TO_PYTHON_GYM_LIBRARY/gym/envs` as follow to
create a new registration entry:

    ```
    register(
        id="LunarLander-v2-mo-custom",
        entry_point="gym.envs.box2d.lunar_lander_mo_custom:LunarLander",
        max_episode_steps=1000,
        reward_threshold=200,
    )
    ```

* Eventually you can import your custom environment in Python or Julia using the 'id' of the registration entry:

    Julia   ->  `my_env = GymEnv("LunarLander-v2-mo-custom")` 
