
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import magnum as mn
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import attr
import magnum as mn
import numpy as np
import quaternion  # noqa: F401

from habitat_sim import registry
from habitat_sim.agent import SceneNodeControl

import cv2


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings['height'], settings['width']]
    depth_sensor_spec.position   = [0.0, settings['sensor_height'], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE


    # Custom action spaces
    # We can also re-register this function such that it effects just the sensors
    habitat_sim.registry.register_move_fn(
        LookUp, name="look_down", body_action=False
    )

    habitat_sim.registry.register_move_fn(
        LookDown, name="look_up", body_action=False
    )

    habitat_sim.registry.register_move_fn(
        MoveBackward, name="move_backward", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        MoveLeft, name="move_left", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        MoveRight, name="move_right", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        TurnLeft, name="turn_left", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        TurnLeft, name="turn_l", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        TurnRight, name="turn_right", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        MoveForward, name="move_forward", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        MoveUp, name="move_up", body_action=True
    )

    habitat_sim.registry.register_move_fn(
        MoveDown, name="move_down", body_action=True
    )

    translation_amount = 0.1
    rotation_amount    = 0.3
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
    agent_cfg.action_space = {
        "move_forward" : habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=translation_amount)),
        "move_backward": habitat_sim.ActionSpec("move_backward",      habitat_sim.agent.ActuationSpec(amount=translation_amount)),
        "move_left"    : habitat_sim.ActionSpec("move_left",          habitat_sim.agent.ActuationSpec(amount=translation_amount)),
        "move_right"   : habitat_sim.ActionSpec("move_right",         habitat_sim.agent.ActuationSpec(amount=translation_amount)),
        "look_down"    : habitat_sim.ActionSpec("look_down",          habitat_sim.agent.ActuationSpec(amount=rotation_amount)),
        "look_up"      : habitat_sim.ActionSpec("look_up",            habitat_sim.agent.ActuationSpec(amount=rotation_amount)),
        "turn_left"    : habitat_sim.ActionSpec("turn_left",          habitat_sim.agent.ActuationSpec(amount=rotation_amount)),
        "turn_l"       : habitat_sim.ActionSpec("turn_l",             habitat_sim.agent.ActuationSpec(amount=rotation_amount)),
        "turn_right"   : habitat_sim.ActionSpec("turn_right",         habitat_sim.agent.ActuationSpec(amount=rotation_amount)),
        "move_up"      : habitat_sim.ActionSpec("move_up",            habitat_sim.agent.ActuationSpec(amount=translation_amount)),
        "move_down"    : habitat_sim.ActionSpec("move_down",          habitat_sim.agent.ActuationSpec(amount=translation_amount)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# This is wrapped in a such that it can be added to a unit test

@attr.s(auto_attribs=True, slots=True)
class look_up_down:
    spin_amount: float

@habitat_sim.registry.register_move_fn(body_action=True)
class LookDown(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: look_up_down
    ):
        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.LEFT
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()


class LookUp(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: look_up_down
    ):

        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.RIGHT
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()


@registry.register_move_fn(body_action=True)

class MoveForward(SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec) -> None:
    
        forward_ax = np.array([0, 0, -0.1])
       
        scene_node.translate_local(forward_ax * actuation_spec.amount)


class MoveUp(SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec) -> None:
    
        forward_ax = np.array([0, 0.1, 0.0])
       
        scene_node.translate_local(forward_ax * actuation_spec.amount)


class MoveDown(SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec) -> None:
    
        forward_ax = np.array([0, -0.1, 0])
       
        scene_node.translate_local(forward_ax * actuation_spec.amount)

class MoveBackward(SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec) -> None:
    
        forward_ax = np.array([0, 0, 0.1])
        
        scene_node.translate_local(forward_ax * actuation_spec.amount)

class MoveLeft(SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec) -> None:

        forward_ax = np.array([-0.1, 0, 0])
    
        scene_node.translate_local(forward_ax * actuation_spec.amount)

class MoveRight(SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: habitat_sim.agent.ActuationSpec) -> None:

        forward_ax = np.array([0.1, 0, 0])
      
        scene_node.translate_local(forward_ax * actuation_spec.amount)


class TurnLeft(SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: look_up_down
    ):
        # Rotate about the +y (up) axis
        rotation_ax = np.array([0.0, 1.0, 0.0])
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()

class TurnRight(SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: look_up_down
    ):
        # Rotate about the +y (up) axis
        rotation_ax = np.array([0.0, -1.0, 0.0])
        scene_node.rotate_local(mn.Deg(actuation_spec.amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

# This is the scene we are going to load.
# we support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
test_scene = "../steve_file.glb"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0,  # Height of sensors in meters, relative to the agent
    "width":  1280,  # Spatial resolution of the observations
    "height": 720,
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

cfg = make_simple_cfg(sim_settings)

# # create simulator
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # in world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        cv2.imshow(action, observations["color_sensor"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


observations = sim.step("turn_left")
# print(observations["color_sensor"].shape, observations['depth_sensor'].shape)
print(observations["color_sensor"].shape)

save = False
save_root = test_scene.split('/')[-1].split('.')[0]
os.makedirs(f"{save_root}/color/", exist_ok=True)
os.makedirs(f"{save_root}/depth/", exist_ok=True)
count = 0

def reverse_action(action):
    if action=='move_forward':
        return 'move_backward'

    elif action =='move_backward':
        return 'move_forward'

    elif action == 'move_up':
        return 'move_down'

    elif action == 'move_down':
        return 'move_up'

    elif action=='move_left':
        return 'move_right'

    elif action=='move_right':
        return 'move_left'

    elif action =='turn_right':
        return 'turn_l'

    elif action == 'turn_l':
        return 'turn_right'

    elif action == 'look_down':
        return 'look_up'

    elif action == 'look_up':
        return 'look_down'

key_actin_dict = {106: "turn_l", 97: "move_left", 108:"turn_right", 100:"move_right", 110:"move_up",109:"move_down",
105:"look_down", 107:"look_up", 119:"move_forward", 115:"move_backward"}
list_of_action = []
while True:

    rgb_img = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
    cv2.imshow('san', rgb_img)
    key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if key != 8:
        list_of_action.append(key)

    print(count, key)
    if key == 106:
        observations = sim.step("turn_l")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)

    elif key == 97:
        observations = sim.step("move_left")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)

    elif key == 108:
        # agent_state.rotation = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        observations  = sim.step("turn_right")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)


    elif key == 100:
        # agent_state.rotation = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        observations  = sim.step("move_right")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)

    elif key == 110:
        # agent_state.rotation = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        observations  = sim.step("move_up")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)

    elif key == 109:
        # agent_state.rotation = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        observations  = sim.step("move_down")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)

    elif key == 105:
        # agent_state.rotation = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        observations  = sim.step("look_down")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)

    elif key == 8:
        print(len(list_of_action))
        take_reverse_action       =  reverse_action(key_actin_dict[list_of_action[-1]])
        print(take_reverse_action)
        observations              = sim.step(take_reverse_action)
        count  = count - 1 

        try:
            os.remove(f"{save_root}/color/{str(count).zfill(7)}.jpg")
            os.remove(f"{save_root}/depth/{str(count).zfill(7)}.png")

        except:
            print(f"NO_files=>{save_root}/color/{str(count).zfill(7)}")
            pass

        list_of_action.pop(-1)
        continue


    elif key == 107:
        # agent_state.rotation = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        observations  = sim.step("look_up")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)


    elif key == 119:
        observations = sim.step("move_forward")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)


    elif key == 115:
        observations = sim.step("move_backward")
        if save:
            rgb_img     = cv2.cvtColor(np.asarray(Image.fromarray(observations["color_sensor"], mode="RGBA")), cv2.COLOR_BGR2RGB)
            depth_image = (observations["depth_sensor"] * 1000).astype(np.uint16)

            cv2.imwrite(f"{save_root}/color/{str(count).zfill(7)}.jpg", rgb_img)
            cv2.imwrite(f"{save_root}/depth/{str(count).zfill(7)}.png", depth_image)


    elif key == 32:
        save = True

    elif key == 113:
        break

    count += 1