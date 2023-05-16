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
