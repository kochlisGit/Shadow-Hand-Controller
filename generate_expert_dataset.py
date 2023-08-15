from controllers.expert import ExpertController
from simulator.pyopengl import GLFWSimulator
from utils import control
from utils import dataset

shadow_hand_xml_filepath = 'objects/shadow_hand/scene_left.xml'
signs_filepath = 'data/signs.json'
ctrl_limits_filepath = 'data/ctrl_limits.csv'
dataset_filepath = 'data/expert_dataset.csv'
trajectory_steps = 100
cam_verbose = False
sim_verbose = True


def main():
    signs = control.read_sign_transitions(json_filepath=signs_filepath)
    ctrl_limits = control.read_ctrl_limits(csv_filepath=ctrl_limits_filepath)

    hand_controller = ExpertController(
        ctrl_limits=ctrl_limits,
        signs=signs
    )
    simulator = GLFWSimulator(
        shadow_hand_xml_filepath=shadow_hand_xml_filepath,
        hand_controller=hand_controller,
        trajectory_steps=trajectory_steps,
        cam_verbose=cam_verbose,
        sim_verbose=sim_verbose
    )
    simulator.run()

    transition_history = simulator.transition_history
    dataset.generate_expert_dataset(transition_history=transition_history, dataset_filepath=dataset_filepath)


if __name__ == '__main__':
    main()
