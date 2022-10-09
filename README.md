
# Create dataset from habitat simulator
This script helps to navigate your 3d scene using keyboard. It also capture rgb image and depth images. You can create your own dataset like matterport using this script


## Installation

Install habitat sim

```bash
conda create -n habitat python=3.7 cmake=3.14.0
conda activate habitat

conda install habitat-sim -c conda-forge -c aihabitat
```
    
## Settings
Change your model file path and settings at 
```
create_dataset_simulator.py
```
```python
test_scene = "File path to your 3d model (.glb, .obj, .ply)"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0,  # Height of sensors in meters, relative to the agent
    "width":  1280,  # Spatial resolution of the observations
    "height": 720,
}
```

## Running Tests

Run following code

```bash
  conda activate habitat
  python create_dataset_simulator.py
```

Use space to start capturing your dataset. The data will be saved only after pressing the "space bar". The output will be saved with the same name of the input model.

Use your keyboard to move in the 3d model:
```bash                   
w - move_forward     
s - move_backward
a - move left
d - move right

i - turn up
k - turn down
j - turn left
k - turn right
```

