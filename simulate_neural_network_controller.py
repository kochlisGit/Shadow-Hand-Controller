from controllers.model import ModelController
from models.tf.nn import NeuralNetwork
from simulator.pyopengl import GLFWSimulator
from utils import control
from utils import dataset

model_checkpoint_directory = 'checkpoints/nn'
shadow_hand_xml_filepath = 'objects/shadow_hand/scene_left.xml'
ctrl_limits_filepath = 'data/ctrl_limits.csv'
dataset_filepath = 'data/expert_dataset.csv'
trajectory_steps = 100
cam_verbose = False
sim_verbose = True


def main():
    model = NeuralNetwork(input_shapes={'sign': (), 'order': ()}, num_outputs=-1)
    model.load(checkpoint_directory=model_checkpoint_directory)

    ctrl_limits = control.read_ctrl_limits(csv_filepath=ctrl_limits_filepath)
    hand_controller = ModelController(
        model=model,
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
