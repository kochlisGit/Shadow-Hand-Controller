import ray
from controllers.model import ModelController
from models.rllib.ppo import PPOAgent
from simulator.pyopengl import GLFWSimulator
from utils import control
from utils import dataset

agent_checkpoint_directory = 'checkpoints/ppo/checkpoint_001000'
shadow_hand_xml_filepath = 'objects/shadow_hand/scene_left.xml'
signs_filepath = 'data/signs.json'
ctrl_limits_filepath = 'data/ctrl_limits.csv'
dataset_filepath = 'data/expert_dataset.csv'
trajectory_steps = 100
cam_verbose = False
sim_verbose = True


def main():
    ray.shutdown()
    ray.init()

    agent = PPOAgent(env_config={}, episode_steps=-1)
    agent.load(checkpoint_directory=agent_checkpoint_directory)

    ctrl_limits = control.read_ctrl_limits(csv_filepath=ctrl_limits_filepath)
    hand_controller = ModelController(
        model=agent,
        ctrl_limits=ctrl_limits,
        num_actuators=dataset.NUM_ACTUATORS,
        one_hot_signs=dataset.ONE_HOT_SIGNS,
        one_hot_orders=dataset.ONE_HOT_ORDERS
    )
    simulator = GLFWSimulator(
        shadow_hand_xml_filepath=shadow_hand_xml_filepath,
        hand_controller=hand_controller,
        trajectory_steps=trajectory_steps,
        cam_verbose=cam_verbose,
        sim_verbose=sim_verbose
    )
    simulator.run()


if __name__ == '__main__':
    main()
