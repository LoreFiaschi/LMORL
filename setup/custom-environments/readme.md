### How to load a custom environment in gym

* In order to create a custom environment you need to put .py environment file inside /home/matteopierucci/.local/lib/python3.10/site-packages/gym/envs/box2d

* Then edit the __init__.py inside /lib/python3.10/site-packages/gym/envs as follow to
create a new registration entry:

    ```register(
        id="LunarLander-v2-mo-custom",
        entry_point="gym.envs.box2d.lunar_lander_mo_custom:LunarLander",
        max_episode_steps=1000,
        reward_threshold=200,
    )```

* Eventually you can import your custom environment in Python or Julia using the 'id' of the registration entry:

    Julia   ->  `my_env = GymEnv("LunarLander-v2-mo-custom")` 
