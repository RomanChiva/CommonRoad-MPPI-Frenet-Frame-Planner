#!/user/bin/env python

"""Sampling-based trajectory planning in a frenet frame considering ethical implications."""

# Standard imports
import os
import sys
import copy
import warnings
import json
from inspect import currentframe, getframeinfo
import pathlib


# Third party imports
import numpy as np
import matplotlib
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
)

from commonroad_helper_functions.sensor_model import get_visible_objects
from commonroad_helper_functions.exceptions import (
    ExecutionTimeoutError,
)
from prediction import WaleNet

# Custom imports
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(mopl_path)

from PA_CommonRoad.planner.planning import Planner
from PA_CommonRoad.planner.utils.timeout import Timeout
from PA_CommonRoad.planner.Frenet.utils.visualization import draw_frenet_trajectories, draw_optimal_trajectory, draw_MPPI_valid
from PA_CommonRoad.planner.Frenet.utils.validity_checks import (
    VALIDITY_LEVELS,
)
from PA_CommonRoad.planner.Frenet.utils.helper_functions import (
    get_goal_area_shape_group,
)
from PA_CommonRoad.planner.Frenet.utils.prediction_helpers import (
    add_static_obstacle_to_prediction,
    get_dyn_and_stat_obstacles,
    get_ground_truth_prediction,
    get_obstacles_in_radius,
    get_orientation_velocity_and_shape_of_prediction,
)
from PA_CommonRoad.planner.Frenet.configs.load_json import (
    load_harm_parameter_json,
    load_planning_json,
    load_risk_json,
    load_weight_json,
)
from PA_CommonRoad.planner.Frenet.utils.frenet_functions import (
    calc_frenet_trajectories,
    get_v_list,
    sort_frenet_trajectories,
    sort_frenet_trajectories_samples,
)
from PA_CommonRoad.planner.Frenet.utils.logging import FrenetLogging
from PA_CommonRoad.planner.utils.responsibility import assign_responsibility_by_action_space
from PA_CommonRoad.planner.utils import reachable_set
from PA_CommonRoad.risk_assessment.visualization.risk_visualization import (
    create_risk_files,
)
from PA_CommonRoad.risk_assessment.visualization.risk_dashboard import risk_dashboard

### Frenet MPPI
from PA_CommonRoad.planner.Frenet.MPPI_Objective_Functions.Frenet_Objective import Objective
### MPPI Imports
from prediction import WaleNet
from mppi_planner.mppi_isaac import MPPIisaacPlanner
from mppi_planner.mppi import MPPIPlanner
from mppi_utils.config_store import ExampleConfig
import hydra
from hydra.experimental import initialize, compose
import torch
from omegaconf import OmegaConf
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import time

from PA_CommonRoad.planner.Frenet.utils.cart2frenet import C2F
from PA_CommonRoad.planner.Frenet.utils.frenet_functions import FrenetTrajectory
from PA_CommonRoad.planner.Frenet.utils.KL_Cost import KL_Cost
import matplotlib.pyplot as plt


class FrenetPlanner_MPPI(Planner):
    """Jerk optimal planning in frenet coordinates with quintic polynomials in lateral direction and quartic polynomials in longitudinal direction."""

    # SAME INITIALIZATION AS IN THE ORIGINAL FrenetPlanner
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        ego_id: int,
        vehicle_params,
        mode,
        exec_timer=None,
        frenet_parameters: dict = None,
        sensor_radius: float = 50.0,
        plot_frenet_trajectories: bool = False,
        weights=None,
        settings=None,
    ):
        """
        Initialize a frenét planner.

        Args:
            scenario (Scenario): Scenario.
            planning_problem (PlanningProblem): Given planning problem.
            ego_id (int): ID of the ego vehicle.
            vehicle_params (VehicleParameters): Parameters of the ego vehicle.
            mode (Str): Mode of the frenét planner.
            timing (bool): True if the execution times should be saved. Defaults to False.
            frenet_parameters (dict): Parameters for the frenét planner. Defaults to None.
            sensor_radius (float): Radius of the sensor model. Defaults to 30.0.
            plot_frenet_trajectories (bool): True if the frenét paths should be visualized. Defaults to False.
            weights(dict): the weights of the costfunction. Defaults to None.
        """
        super().__init__(scenario, planning_problem, ego_id, vehicle_params, exec_timer)


        # Set up logger
        self.logger = FrenetLogging(
            log_path=f"./planner/Frenet/results/logs/{scenario.benchmark_id}.csv"
        )

        try:
            with Timeout(10, "Frenet Planner initialization"):

                self.exec_timer.start_timer("initialization/total")
                if frenet_parameters is None:
                    print(
                        "No frenet parameters found. Swichting to default parameters."
                    )
                    frenet_parameters = {
                        "t_list": [2.0],
                        "v_list_generation_mode": "linspace",
                        "n_v_samples": 5,
                        "d_list": np.linspace(-3.5, 3.5, 15),
                        "dt": 0.1,
                        "v_thr": 3.0,
                    }

                # parameters for frenet planner
                self.frenet_parameters = frenet_parameters
                # vehicle parameters
                self.p = vehicle_params

                # load parameters
                self.params_harm = load_harm_parameter_json()
                if weights is None:
                    self.params_weights = load_weight_json()
                else:
                    self.params_weights = weights
                if settings is not None:
                    if "risk_dict" in settings:
                        self.params_mode = settings["risk_dict"]
                    else:
                        self.params_mode = load_risk_json()

                self.params_dict = {
                    'weights': self.params_weights,
                    'modes': self.params_mode,
                    'harm': self.params_harm,
                }

                # check if the planning problem has a goal velocity and safe it
                if hasattr(planning_problem.goal.state_list[0], "velocity"):
                    self.v_goal_min = planning_problem.goal.state_list[0].velocity.start
                    self.v_goal_max = planning_problem.goal.state_list[0].velocity.end
                else:
                    self.v_goal_min = None
                    self.v_goal_max = None

                self.cost_dict = {}

                # check if the planning problem has an initial acceleration, else set it to zero
                if not hasattr(self.ego_state, "acceleration"):
                    self.ego_state.acceleration = 0.0

                # initialize the driven trajectory with the initial position
                self.driven_traj = [
                    State(
                        position=self.ego_state.position,
                        orientation=self.ego_state.orientation,
                        time_step=self.ego_state.time_step,
                        velocity=self.ego_state.velocity,
                        acceleration=self.ego_state.acceleration,
                    )
                ]

                # get sensor radius param, and planner mode
                self.sensor_radius = sensor_radius
                self.mode = mode

                # get visualization marker
                self.plot_frenet_trajectories = plot_frenet_trajectories

                ### LOAD THE MPPI COnfigutration
                config = self._load_config()
                self.cfg = OmegaConf.to_object(config)



                # initialize the prediction network if necessary
                if self.mode == "WaleNet" or self.mode == "risk":

                    prediction_config_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "configs",
                        "prediction.json",
                    )
                    with open(prediction_config_path, "r") as f:
                        online_args = json.load(f)

                    self.predictor = WaleNet(scenario=scenario, online_args=online_args, verbose=False)
                elif self.mode == "ground_truth":
                    self.predictor = None
                else:
                    raise ValueError("mode must be ground_truth, WaleNet, or risk")

                # check whether reachable sets have to be calculated for responsibility
                if (
                    'responsibility' in self.params_weights
                    and self.params_weights['responsibility'] > 0
                ):
                    self.responsibility = True
                    self.reach_set = reachable_set.ReachSet(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_length=self.p.l,
                        ego_width=self.p.w,
                    )
                else:
                    self.responsibility = False
                    self.reach_set = None

                # create a collision object of the non-lanelet area of the scenario to check if a trajectory leaves the road
                # for some scenarios, this does not work/takes forever
                # to avoid that, abort after 5 seconds and raise an error
                with self.exec_timer.time_with_cm(
                    "initialization/initialize road boundary"
                ):
                    try:
                        with Timeout(5, "Initializing roud boundary"):
                            (
                                _,
                                self.road_boundary,
                            ) = boundary.create_road_boundary_obstacle(
                                scenario=scenario,
                                method="aligned_triangulation",
                                axis=2,
                            )
                    except ExecutionTimeoutError:
                        raise RuntimeError("Road Boundary can not be created")

                # create a collision checker
                # remove the ego vehicle from the scenario
                with self.exec_timer.time_with_cm(
                    "initialization/initialize collision checker"
                ):
                    cc_scenario = copy.deepcopy(self.scenario)
                    cc_scenario.remove_obstacle(
                        obstacle=[cc_scenario.obstacle_by_id(ego_id)]
                    )
                    try:
                        self.collision_checker = create_collision_checker(cc_scenario)
                    except Exception:
                        raise BrokenPipeError("Collision Checker fails.") from None

                with self.exec_timer.time_with_cm(
                    "initialization/initialize goal area"
                ):
                    # get the shape group of the goal area
                    self.goal_area = get_goal_area_shape_group(
                        planning_problem=self.planning_problem, scenario=self.scenario
                    )

                self.exec_timer.stop_timer("initialization/total")
        except ExecutionTimeoutError:
            raise TimeoutError


    # Load the requred MPPI Settings
    def _load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../conf"):
            config = compose(config_name="config_bmw")
        return config

    # Create the Planner (Wapper around the core MPPI Class to ease interactions, like an interface)
    def _set_planner(self, cfg, prediction):
        """
        Initializes the mppi planner for jackal robot.

        Params
        ----------
        goal_position: np.ndarray
            The goal to the motion planning problem.
        """


        objective = Objective(cfg, cfg.mppi.device, self.reference_spline, self.global_path, prediction)
        mppi_planner = MPPIisaacPlanner(cfg, objective)

        return mppi_planner

    
    def _step_planner(self):

        """Frenet Planner step function. Generate optimal trajectory using MPPI as opposed to the splines"""

        self.exec_timer.start_timer("simulation/total")
    

        ##############################################################
        # get the current state of the ego vehicle and the time step
        ################################################################
        with self.exec_timer.time_with_cm("simulation/update driven trajectory"):
            # update the driven trajectory
            # add the current state to the driven path

            if self.ego_state.time_step != 0:
                current_state = State(
                    position=self.ego_state.position,
                    orientation=self.ego_state.orientation,
                    time_step=self.ego_state.time_step,
                    velocity=self.ego_state.velocity,
                    acceleration=self.ego_state.acceleration,
                )
                
                self.driven_traj.append(current_state)
        ################################################################
        # Get the predictions
        ################################################################



        t_list = self.frenet_parameters["t_list"]
        with self.exec_timer.time_with_cm("simulation/prediction"):
            # Overwrite later
            visible_area = None
            # get visible objects if the prediction is used
            if self.mode == "WaleNet" or self.mode == "risk":
                # get_visible_objects may fail sometimes due to bad lanelets (e.g. DEU_A9-1_1_T-1 at [-73.94, -53.24])
                if self.params_mode["sensor_occlusion_model"]:
                    try:
                        visible_obstacles, visible_area = get_visible_objects(
                            scenario=self.scenario,
                            ego_pos=self.ego_state.position,
                            time_step=self.time_step,
                            sensor_radius=self.sensor_radius,
                        )
                    except Exception as e:  # TopologicalError or AttributeError:
                        # if get_visible_objects fails just get every obstacle in the sensor_radius
                        print(
                            f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                            e,
                        )
                        visible_obstacles = get_obstacles_in_radius(
                            scenario=self.scenario,
                            ego_id=self.ego_id,
                            ego_state=self.ego_state,
                            radius=self.sensor_radius,
                        )
                else:
                    
                    visible_obstacles = get_obstacles_in_radius(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_state=self.ego_state,
                        radius=self.sensor_radius,
                    )

                # predictions may fail (e.g. SetBasedPrediction DEU_Ffb-1_2_S-1)
                try:
                    # get dynamic and static visible obstacles since predictor can not handle static obstacles
                    (
                        dyn_visible_obstacles,
                        stat_visible_obstacles,
                    ) = get_dyn_and_stat_obstacles(
                        scenario=self.scenario, obstacle_ids=visible_obstacles
                    )


                    # Hack for predictions with other Ego Agents (Full trajectory not available resort to this)
                    # Investigate maybe doing this at timestep 0 (Might freak out otherwise?)
                    for obstacle in dyn_visible_obstacles:

                        if len(self.scenario._dynamic_obstacles[obstacle].prediction.trajectory.state_list) <= self.ego_state.time_step:
                            print('Duplicated State at timestep:', self.ego_state.time_step)
                            self.scenario._dynamic_obstacles[obstacle].prediction.trajectory.state_list.append(
                                self.scenario._dynamic_obstacles[obstacle].prediction.trajectory.state_list[-1]
                            )



                    # get prediction for dynamic obstacles
                    predictions = self.predictor.step(
                        time_step=self.ego_state.time_step,
                        obstacle_id_list=dyn_visible_obstacles,
                        scenario=self.scenario,
                    )
                    #create and add prediction of static obstacles
                    predictions = add_static_obstacle_to_prediction(
                        scenario=self.scenario,
                        predictions=predictions,
                        obstacle_id_list=stat_visible_obstacles,
                        pred_horizon=max(t_list) / self.scenario.dt,
                    )
                    
                # if prediction fails use ground truth as prediction
                except Exception as e:
                    print(
                        f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                        e,
                    )
                    predictions = get_ground_truth_prediction(
                        scenario=self.scenario,
                        obstacle_ids=visible_obstacles,
                        time_step=self.ego_state.time_step,
                    )


                # add orientation and dimensions of the obstacles to the prediction
                predictions = get_orientation_velocity_and_shape_of_prediction(
                    predictions=predictions, scenario=self.scenario
                )
                
                # Assign responsibility to predictions
                predictions = assign_responsibility_by_action_space(
                    self.scenario, self.ego_state, predictions
                )
                

            else:
                # TODO: Get GT prediction here for responsibility
                predictions = None
        # calculate reachable sets
        if self.responsibility:
            with self.exec_timer.time_with_cm(
                    "simulation/calculate and check reachable sets"
            ):
                self.reach_set.calc_reach_sets(self.ego_state, list(predictions.keys()))

            if self.reach_set is not None:
                log_reach_set = self.reach_set.reach_sets[self.time_step]
            else:
                log_reach_set = None
        
        #########################################################################
        # Generate MPPI TRAJECTORIES
        ########################################################################

        mppi_planner = self._set_planner(self.cfg, predictions)
        


        # Get all the encessary attributes from the ego state (Steering angle only becomes available later on (We create it ourselves))
        try:
            steering_angle = self.ego_state.steering_angle
            
        except:
            steering_angle = 0.0
        

 
        vehicle_state = [self.ego_state.position[0], self.ego_state.position[1], self.ego_state.orientation, self.ego_state.velocity, steering_angle]
        vehicle_velocity = [self.ego_state.velocity]
        print('Velocity Ego:', vehicle_velocity)

        # Generate trajectories
        actions, states = mppi_planner.get_samples(q=vehicle_state, qdot=vehicle_velocity)
       

       # Visualize MPPI Trajectories plot X, Y 
        # Plot in a new figure an stop execution
        
        #######################################################################
        # Convert them to Frenet Frame
        #######################################################################

        # Convert the trajectories to frenet frame and fit them to the trajectory list format
        start_time = time.time()
        states_numpy = states.cpu().numpy()
        s, d = mppi_planner.convert_to_frenet(states)

        # Get the frenet trajectories
        ft_list_1 = C2F(s,d, states_numpy, self.frenet_parameters["dt"], self.p.w/0.1)
        end_time = time.time()
        

        #######################################################################
        # Check validity of the trajectories, make sure no constraints are broken: 

        # Check for: 
            # 1. Physically Possible with car dynamics
            # 2. Collision Free
            # 3. In the road boundary
            # 4. Check RISK constraints are satisfied
            # It all good the trajectory is valid
        ###########################################################################
       
        # # Sort the trajectories
        # start_time = time.time()
        # with self.exec_timer.time_with_cm("simulation/sort trajectories/total"):
        #     # sorted list (increasing costs)
        #     ft_list_valid, ft_list_invalid, validity_dict = sort_frenet_trajectories(
        #         ego_state=self.ego_state,
        #         fp_list=ft_list_1,
        #         global_path=self.global_path,
        #         predictions=predictions,
        #         mode=self.mode,
        #         params=self.params_dict,
        #         planning_problem=self.planning_problem,
        #         scenario=self.scenario,
        #         vehicle_params=self.p,
        #         ego_id=self.ego_id,
        #         dt=self.frenet_parameters["dt"],
        #         sensor_radius=self.sensor_radius,
        #         road_boundary=self.road_boundary,
        #         collision_checker=self.collision_checker,
        #         goal_area=self.goal_area,
        #         exec_timer=self.exec_timer,
        #         reach_set=(self.reach_set if self.responsibility else None)
        #     )
        # end_time = time.time()
        # print('Time taken to sort frenet trajectories:', end_time - start_time)
        
        # print('Number of valid trajectories:', len(ft_list_valid))

        # # Get invalid trajectories
        # drop = [ft.traj_index for ft in ft_list_invalid]
        # reason_invalid = [ft.reason_invalid for ft in ft_list_invalid]

        # # Plot the trajectories
        # # Make both axes the same scale
        # fig, ax = plt.subplots()
        # ax.set_aspect('equal', 'box')
        # for i in range(states.size(0)):
        #     if i in drop:
        #         ax.plot(states[i,:, 0], states[i,:, 1], 'r')
        #     else:
        #         ax.plot(states[i,:, 0], states[i,:, 1], 'b')
        
    

        #######################################################################
        # Pass the remaining valid trajectories to the MPPI Planner to calculate cost and find an optimal trajectory
        ###########################################################################

        ft_list_valid = ft_list_1
        ft_list_invalid = []
        drop = []

        action, plan, all_trajs, COST = mppi_planner.compute_traj(drop)

        # Make a histogram of the costs and plot it in a separate window
        figz,axz = plt.subplots()
        axz.hist(COST, bins=50)
        axz.set_title('Cost Distribution')
        axz.set_xlabel('Cost')
        axz.set_ylabel('Frequency')
        plt.show()
        



        # Transform cost with MPPI boltzmann
        # FInd minimum
        min_cost = min(COST)
        COST = [np.exp(-self.cfg.mppi.lambda_*(cost - min_cost)) for cost in COST]
        # Normalize
        
        # FOr each trajectory in valid assign its cost
        for i in range(len(ft_list_valid)):
            ft_list_valid[i].cost = COST[i]
        



        # Mapping tensor columns to dictionary keys
        column_mapping = {
            0: 'x_m',
            1: 'y_m',
            2: 'psi_rad',
            3: 'v_mps',
            4: 'delta_rad',
        }
        # Print _trajectory keys
        # And types
        
        # Filling in values from the tensor into the dictionary arrays
        for col_index, key in column_mapping.items():
            self._trajectory[key][:] = plan[:, col_index].numpy()
        
        self._trajectory['acceleration_mps2'][:] = action[:, 0].numpy()
        

      
      #######################################################################
      # RENDER THE OPTIMAL TRAJECTORY
      ###########################################################################
        
        # with self.exec_timer.time_with_cm("plot trajectories"):
        #     # print some information about the optimal trajectory
        #     matplotlib.use("TKAgg")
        #     try:
        #         all_trajs = all_trajs.cpu().numpy()
        #         draw_optimal_trajectory(
        #             scenario=self.scenario,
        #             time_step=self.ego_state.time_step,
        #             marked_vehicle=self.ego_id,
        #             planning_problem=self.planning_problem,
        #             all_trajs=all_trajs,
        #             traj=self._trajectory,
        #             global_path=self.global_path_to_goal,
        #             global_path_after_goal=self.global_path_after_goal,
        #             driven_traj=self.driven_traj,
        #             animation_area=50.0,
        #             predictions=predictions,
        #             visible_area=visible_area,
        #         )
           

        #     except Exception as e:
        #         print(e)

        with self.exec_timer.time_with_cm("plot trajectories"):
            # print some information about the optimal trajectory
            matplotlib.use("TKAgg")
            try:
                draw_MPPI_valid(
                    scenario=self.scenario,
                    time_step=self.ego_state.time_step,
                    marked_vehicle=self.ego_id,
                    planning_problem=self.planning_problem,
                    VALID_traj=ft_list_valid,
                    INVALID_traj=ft_list_invalid,
                    traj=self._trajectory,
                    global_path=self.global_path_to_goal,
                    global_path_after_goal=self.global_path_after_goal,
                    driven_traj=self.driven_traj,
                    animation_area=50.0,
                    predictions=predictions,
                    visible_area=visible_area,
                )
           

            except Exception as e:
                print('ERROR PLOTTING')
                print(e)
        


    #######################################################################
    # DONE!
    ###########################################################################





class FrenetPlanner_SAMPLES(Planner):
    """Jerk optimal planning in frenet coordinates with quintic polynomials in lateral direction and quartic polynomials in longitudinal direction."""

    # SAME INITIALIZATION AS IN THE ORIGINAL FrenetPlanner
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        ego_id: int,
        vehicle_params,
        mode,
        exec_timer=None,
        frenet_parameters: dict = None,
        sensor_radius: float = 50.0,
        plot_frenet_trajectories: bool = False,
        weights=None,
        settings=None,
    ):
        """
        Initialize a frenét planner.

        Args:
            scenario (Scenario): Scenario.
            planning_problem (PlanningProblem): Given planning problem.
            ego_id (int): ID of the ego vehicle.
            vehicle_params (VehicleParameters): Parameters of the ego vehicle.
            mode (Str): Mode of the frenét planner.
            timing (bool): True if the execution times should be saved. Defaults to False.
            frenet_parameters (dict): Parameters for the frenét planner. Defaults to None.
            sensor_radius (float): Radius of the sensor model. Defaults to 30.0.
            plot_frenet_trajectories (bool): True if the frenét paths should be visualized. Defaults to False.
            weights(dict): the weights of the costfunction. Defaults to None.
        """
        super().__init__(scenario, planning_problem, ego_id, vehicle_params, exec_timer)


        # Set up logger
        self.logger = FrenetLogging(
            log_path=f"./planner/Frenet/results/logs/{scenario.benchmark_id}.csv"
        )

        try:
            with Timeout(10, "Frenet Planner initialization"):

                self.exec_timer.start_timer("initialization/total")
                if frenet_parameters is None:
                    print(
                        "No frenet parameters found. Swichting to default parameters."
                    )
                    frenet_parameters = {
                        "t_list": [2.0],
                        "v_list_generation_mode": "linspace",
                        "n_v_samples": 5,
                        "d_list": np.linspace(-3.5, 3.5, 15),
                        "dt": 0.1,
                        "v_thr": 3.0,
                    }

                # parameters for frenet planner
                self.frenet_parameters = frenet_parameters
                # vehicle parameters
                self.p = vehicle_params

                # load parameters
                self.params_harm = load_harm_parameter_json()
                if weights is None:
                    self.params_weights = load_weight_json()
                else:
                    self.params_weights = weights
                if settings is not None:
                    if "risk_dict" in settings:
                        self.params_mode = settings["risk_dict"]
                    else:
                        self.params_mode = load_risk_json()

                self.params_dict = {
                    'weights': self.params_weights,
                    'modes': self.params_mode,
                    'harm': self.params_harm,
                }

                # check if the planning problem has a goal velocity and safe it
                if hasattr(planning_problem.goal.state_list[0], "velocity"):
                    self.v_goal_min = planning_problem.goal.state_list[0].velocity.start
                    self.v_goal_max = planning_problem.goal.state_list[0].velocity.end
                else:
                    self.v_goal_min = None
                    self.v_goal_max = None

                self.cost_dict = {}

                # check if the planning problem has an initial acceleration, else set it to zero
                if not hasattr(self.ego_state, "acceleration"):
                    self.ego_state.acceleration = 0.0

                # initialize the driven trajectory with the initial position
                self.driven_traj = [
                    State(
                        position=self.ego_state.position,
                        orientation=self.ego_state.orientation,
                        time_step=self.ego_state.time_step,
                        velocity=self.ego_state.velocity,
                        acceleration=self.ego_state.acceleration,
                    )
                ]

                # get sensor radius param, and planner mode
                self.sensor_radius = sensor_radius
                self.mode = mode

                # get visualization marker
                self.plot_frenet_trajectories = plot_frenet_trajectories

                ### LOAD THE MPPI COnfigutration
                config = self._load_config()
                self.cfg = OmegaConf.to_object(config)



                # initialize the prediction network if necessary
                if self.mode == "WaleNet" or self.mode == "risk":

                    prediction_config_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "configs",
                        "prediction.json",
                    )
                    with open(prediction_config_path, "r") as f:
                        online_args = json.load(f)

                    self.predictor = WaleNet(scenario=scenario, online_args=online_args, verbose=False)
                elif self.mode == "ground_truth":
                    self.predictor = None
                else:
                    raise ValueError("mode must be ground_truth, WaleNet, or risk")

                # check whether reachable sets have to be calculated for responsibility
                if (
                    'responsibility' in self.params_weights
                    and self.params_weights['responsibility'] > 0
                ):
                    self.responsibility = True
                    self.reach_set = reachable_set.ReachSet(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_length=self.p.l,
                        ego_width=self.p.w,
                    )
                else:
                    self.responsibility = False
                    self.reach_set = None

                # create a collision object of the non-lanelet area of the scenario to check if a trajectory leaves the road
                # for some scenarios, this does not work/takes forever
                # to avoid that, abort after 5 seconds and raise an error
                with self.exec_timer.time_with_cm(
                    "initialization/initialize road boundary"
                ):
                    try:
                        with Timeout(5, "Initializing roud boundary"):
                            (
                                _,
                                self.road_boundary,
                            ) = boundary.create_road_boundary_obstacle(
                                scenario=scenario,
                                method="aligned_triangulation",
                                axis=2,
                            )
                    except ExecutionTimeoutError:
                        raise RuntimeError("Road Boundary can not be created")

                # create a collision checker
                # remove the ego vehicle from the scenario
                with self.exec_timer.time_with_cm(
                    "initialization/initialize collision checker"
                ):
                    cc_scenario = copy.deepcopy(self.scenario)
                    cc_scenario.remove_obstacle(
                        obstacle=[cc_scenario.obstacle_by_id(ego_id)]
                    )
                    try:
                        self.collision_checker = create_collision_checker(cc_scenario)
                    except Exception:
                        raise BrokenPipeError("Collision Checker fails.") from None

                with self.exec_timer.time_with_cm(
                    "initialization/initialize goal area"
                ):
                    # get the shape group of the goal area
                    self.goal_area = get_goal_area_shape_group(
                        planning_problem=self.planning_problem, scenario=self.scenario
                    )

                self.exec_timer.stop_timer("initialization/total")
        except ExecutionTimeoutError:
            raise TimeoutError


    # Load the requred MPPI Settings
    def _load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../conf"):
            config = compose(config_name="config_bmw")
        return config

    # Create the Planner (Wapper around the core MPPI Class to ease interactions, like an interface)
    def _set_planner(self, cfg, prediction):
        """
        Initializes the mppi planner for jackal robot.

        Params
        ----------
        goal_position: np.ndarray
            The goal to the motion planning problem.
        """


        self.objective = Objective(cfg, cfg.mppi.device, self.reference_spline, self.global_path, prediction)
        mppi_planner = MPPIisaacPlanner(cfg, self.objective)

        return mppi_planner

    
    def _step_planner(self):

        """Frenet Planner step function. Generate optimal trajectory using MPPI as opposed to the splines"""

        self.exec_timer.start_timer("simulation/total")
    

        ##############################################################
        # get the current state of the ego vehicle and the time step
        ################################################################
        with self.exec_timer.time_with_cm("simulation/update driven trajectory"):
            # update the driven trajectory
            # add the current state to the driven path

            if self.ego_state.time_step != 0:
                current_state = State(
                    position=self.ego_state.position,
                    orientation=self.ego_state.orientation,
                    time_step=self.ego_state.time_step,
                    velocity=self.ego_state.velocity,
                    acceleration=self.ego_state.acceleration,
                )
                
                self.driven_traj.append(current_state)
        ################################################################
        # Get the predictions
        ################################################################


        t_list = self.frenet_parameters["t_list"]
        with self.exec_timer.time_with_cm("simulation/prediction"):
            # Overwrite later
            visible_area = None
            # get visible objects if the prediction is used
            if self.mode == "WaleNet" or self.mode == "risk":
                # get_visible_objects may fail sometimes due to bad lanelets (e.g. DEU_A9-1_1_T-1 at [-73.94, -53.24])
                if self.params_mode["sensor_occlusion_model"]:
                    try:
                        visible_obstacles, visible_area = get_visible_objects(
                            scenario=self.scenario,
                            ego_pos=self.ego_state.position,
                            time_step=self.time_step,
                            sensor_radius=self.sensor_radius,
                        )
                    except Exception as e:  # TopologicalError or AttributeError:
                        # if get_visible_objects fails just get every obstacle in the sensor_radius
                        print(
                            f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                            e,
                        )
                        visible_obstacles = get_obstacles_in_radius(
                            scenario=self.scenario,
                            ego_id=self.ego_id,
                            ego_state=self.ego_state,
                            radius=self.sensor_radius,
                        )
                else:
                    
                    visible_obstacles = get_obstacles_in_radius(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_state=self.ego_state,
                        radius=self.sensor_radius,
                    )

                # predictions may fail (e.g. SetBasedPrediction DEU_Ffb-1_2_S-1)
                try:
                    # get dynamic and static visible obstacles since predictor can not handle static obstacles
                    (
                        dyn_visible_obstacles,
                        stat_visible_obstacles,
                    ) = get_dyn_and_stat_obstacles(
                        scenario=self.scenario, obstacle_ids=visible_obstacles
                    )


                    # Hack for predictions with other Ego Agents (Full trajectory not available resort to this)
                    # Investigate maybe doing this at timestep 0 (Might freak out otherwise?)
                    for obstacle in dyn_visible_obstacles:

                        if len(self.scenario._dynamic_obstacles[obstacle].prediction.trajectory.state_list) <= self.ego_state.time_step:
                            print('Duplicated State at timestep:', self.ego_state.time_step)
                            self.scenario._dynamic_obstacles[obstacle].prediction.trajectory.state_list.append(
                                self.scenario._dynamic_obstacles[obstacle].prediction.trajectory.state_list[-1]
                            )


                    if len(self.scenario._dynamic_obstacles[self.ego_id].prediction.trajectory.state_list) <= self.ego_state.time_step:
                            print('Duplicated Ego at timestep:', self.ego_state.time_step)
                            self.scenario._dynamic_obstacles[self.ego_id].prediction.trajectory.state_list.append(
                                self.scenario._dynamic_obstacles[self.ego_id].prediction.trajectory.state_list[-1])

                    # get prediction for dynamic obstacles
                    predictions = self.predictor.step(
                        time_step=self.ego_state.time_step,
                        obstacle_id_list=dyn_visible_obstacles,
                        scenario=self.scenario,
                    )

                    ego_prediction = self.predictor.step(
                        time_step=self.ego_state.time_step,
                        obstacle_id_list=[self.ego_id],
                        scenario=self.scenario,
                    )
                    ego_prediction = list(ego_prediction.values())[0]
                    # Horizon
                    
                    horizon = int(max(t_list) / self.scenario.dt)
                    ego_pred_pos = ego_prediction['pos_list'][:horizon]
                    ego_pred_cov = ego_prediction['cov_list'][:horizon]


                    #create and add prediction of static obstacles
                    predictions = add_static_obstacle_to_prediction(
                        scenario=self.scenario,
                        predictions=predictions,
                        obstacle_id_list=stat_visible_obstacles,
                        pred_horizon=max(t_list) / self.scenario.dt,
                    )
                    
                # if prediction fails use ground truth as prediction
                except Exception as e:
                    print(
                        f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                        e,
                    )
                    predictions = get_ground_truth_prediction(
                        scenario=self.scenario,
                        obstacle_ids=visible_obstacles,
                        time_step=self.ego_state.time_step,
                    )


                # add orientation and dimensions of the obstacles to the prediction
                predictions = get_orientation_velocity_and_shape_of_prediction(
                    predictions=predictions, scenario=self.scenario
                )
        
                # Assign responsibility to predictions
                predictions = assign_responsibility_by_action_space(
                    self.scenario, self.ego_state, predictions
                )

            else:
                # TODO: Get GT prediction here for responsibility
                predictions = None

        
        # calculate reachable sets
        if self.responsibility:
            with self.exec_timer.time_with_cm(
                    "simulation/calculate and check reachable sets"
            ):
                
                self.reach_set.calc_reach_sets(self.ego_state, list(predictions.keys()))

            if self.reach_set is not None:
                log_reach_set = self.reach_set.reach_sets[self.time_step]
            else:
                log_reach_set = None
        
        #########################################################################
        # Generate Trajectories
        ########################################################################
        

        # Get all the encessary attributes from the ego state (Steering angle only becomes available later on (We create it ourselves))
        try:
            steering_angle = self.ego_state.steering_angle
            
        except:
            steering_angle = 0.0
        
 
        vehicle_state = [self.ego_state.position[0], self.ego_state.position[1], self.ego_state.orientation, self.ego_state.velocity, steering_angle]
        vehicle_velocity = [self.ego_state.velocity]
        print('Velocity Ego:', vehicle_velocity)


         # find position along the reference spline (s, s_d, s_dd, d, d_d, d_dd)
        
        c_s = self.trajectory["s_loc_m"][1]
        c_s_d = self.ego_state.velocity
        c_s_dd = self.ego_state.acceleration
        c_d = self.trajectory["d_loc_m"][1]
        c_d_d = self.trajectory["d_d_loc_mps"][1]
        c_d_dd = self.trajectory["d_dd_loc_mps2"][1]


        # get the end velocities for the frenét paths
        current_v = self.ego_state.velocity
        max_acceleration = self.p.longitudinal.a_max
        t_min = min(self.frenet_parameters["t_list"])
        t_max = max(self.frenet_parameters["t_list"])
        max_v = min(current_v + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max)
        min_v = max(0.01, current_v - max_acceleration * t_min)


        with self.exec_timer.time_with_cm("simulation/get v list"):
            v_list = get_v_list(
                v_min=min_v,
                v_max=max_v,
                v_cur=current_v,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
            )

        with self.exec_timer.time_with_cm("simulation/calculate trajectories/total"):
            d_list = self.frenet_parameters["d_list"]
            t_list = self.frenet_parameters["t_list"]

            # calculate all possible frenét trajectories
            ft_list = calc_frenet_trajectories(
                c_s=c_s,
                c_s_d=c_s_d,
                c_s_dd=c_s_dd,
                c_d=c_d,
                c_d_d=c_d_d,
                c_d_dd=c_d_dd,
                d_list=d_list,
                t_list=t_list,
                v_list=v_list,
                dt=self.frenet_parameters["dt"],
                csp=self.reference_spline,
                v_thr=self.frenet_parameters["v_thr"],
                exec_timer=self.exec_timer,
            )

        #######################################################################
        # Check validity of the trajectories, make sure no constraints are broken: 

        # Check for: 
            # 1. Physically Possible with car dynamics
            # 2. Collision Free
            # 3. In the road boundary
            # 4. Check RISK constraints are satisfied
            # It all good the trajectory is valid
        ###########################################################################
        
       
        # Sort the trajectories
        start_time = time.time()
        with self.exec_timer.time_with_cm("simulation/sort trajectories/total"):
            # sorted list (increasing costs)
            ft_list_valid, ft_list_invalid, validity_dict = sort_frenet_trajectories_samples(
                ego_state=self.ego_state,
                fp_list=ft_list,
                global_path=self.global_path,
                predictions=predictions,
                mode=self.mode,
                params=self.params_dict,
                planning_problem=self.planning_problem,
                scenario=self.scenario,
                vehicle_params=self.p,
                ego_id=self.ego_id,
                dt=self.frenet_parameters["dt"],
                sensor_radius=self.sensor_radius,
                road_boundary=self.road_boundary,
                collision_checker=self.collision_checker,
                goal_area=self.goal_area,
                exec_timer=self.exec_timer,
                reach_set=(self.reach_set if self.responsibility else None)
            )
        end_time = time.time()

        print('Time taken to sort frenet trajectories:', end_time - start_time)
        print('Number of valid trajectories:', len(ft_list_valid))
        print('N Valid:', len(ft_list_valid), 'N Invalid:', len(ft_list_invalid), 'Reasons:', [fp.reason_invalid for fp in ft_list_invalid])

        #######################################################################
        # Convert to Cartesian and compute KL Cost
        ###########################################################################

        # Conver to states tensor shape: N_samples, Horizon, states (X,Y)

        states = [[ft.x.tolist(), ft.y.tolist()] for ft in ft_list_valid]
        states = torch.tensor(states, dtype=torch.float32).to(self.cfg.mppi.device)
        # Switch last two dimensions
        states = states.permute(0,2,1)
        kl_cost = KL_Cost(states, ego_pred_pos, ego_pred_cov, 0.3, 100, discount=0.9)
        kl_cost = kl_cost.tolist()
        # Find index of minimum
        min_cost_index = kl_cost.index(min(kl_cost))

        factor = 0.0

        # Add the KL costs to the trajectory costs
        for i in range(len(ft_list_valid)):
            ft_list_valid[i].cost = ft_list_valid[i].cost + kl_cost[i]*factor

        # Plot the trajectories
        plot = False
        if plot:
        
            fig, ax = plt.subplots()
            ax.set_aspect('equal', 'box')

            # Make colormap with KL costs
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=min(kl_cost), vmax=max(kl_cost))

            for i in range(states.size(0)):
                ax.plot(states[i,:, 0], states[i,:, 1], color = cmap(norm(kl_cost[i])))
            

            # Plot minimum cost with thick red line
            ax.plot(states[min_cost_index,:, 0], states[min_cost_index,:, 1], 'red', linewidth=4)


            # Plot prediction with thicker line
            ax.plot(ego_pred_pos[:,0], ego_pred_pos[:,1], 'black', linewidth=4)
            ax.grid()
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm)
            plt.show()


        #############################################################################
        # Select Optimal Trajectory and set it as _trajectory that will be executed
        ###############################################################################

        costs = [ft.cost for ft in ft_list_valid]

        # Find index of minimum in list
        min_cost_index = costs.index(min(costs))
  
        optimal_trajectory = ft_list_valid[min_cost_index]

        # Set optimal trajectory to the _trajectory

        self._trajectory['x_m'] = optimal_trajectory.x
        self._trajectory['y_m'] = optimal_trajectory.y
        self._trajectory['psi_rad'] = optimal_trajectory.yaw
        self._trajectory['v_mps'] = optimal_trajectory.v
        self._trajectory['delta_rad'] = np.arctan(self.p.l*optimal_trajectory.curv)
        self._trajectory['acceleration_mps2'] = optimal_trajectory.s_dd
        self._trajectory['s_loc_m'] = optimal_trajectory.s
        self._trajectory['d_loc_m'] = optimal_trajectory.d
        self._trajectory['d_d_loc_mps'] = optimal_trajectory.d_d
        self._trajectory['d_dd_loc_mps2'] = optimal_trajectory.d_dd


      
      #######################################################################
      # RENDER THE OPTIMAL TRAJECTORY
      ###########################################################################
        

        # with self.exec_timer.time_with_cm("plot trajectories"):
        #     # print some information about the optimal trajectory
        #     matplotlib.use("TKAgg")
        #     try:
        #         draw_frenet_trajectories(
        #             scenario=self.scenario,
        #             time_step=self.ego_state.time_step,
        #             marked_vehicle=self.ego_id,
        #             planning_problem=self.planning_problem,
        #             traj=optimal_trajectory,
        #             all_traj=ft_list,
        #             global_path=self.global_path_to_goal,
        #             global_path_after_goal=self.global_path_after_goal,
        #             driven_traj=self.driven_traj,
        #             animation_area=50.0,
        #             predictions=predictions,
        #             visible_area=visible_area,
        #         )
        #     except Exception as e:
        #         print(e)
       


    #######################################################################
    # DONE!
    ###########################################################################




if __name__ == "__main__":

    import argparse
    from planner.plannertools.evaluate import ScenarioEvaluator
    from planner.Frenet.plannertools.frenetcreator import FrenetCreator

    parser = argparse.ArgumentParser()
    path ='recorded/SUMO/DEU_Flensburg-32_1_T-1.xml'
    path2 =  "recorded/hand-crafted/ZAM_Tjunction-1_223_T-1.xml"
    path3 = "recorded/hand-crafted/ZAM_Tjunction-1_1_T-1.xml"
    path4 = "recorded/hand-crafted/ZAM_Zip-1_57_T-1.xml"
    path5 = 'recorded/hand-crafted/ZAM_Urban-3_1.xml'
    path6 = 'recorded/cooperative/C-DEU_B471-1_1_T-1.xml'
    path7 = 'recorded/cooperative/C-DEU_B471-2_1.xml'
    path8 = 'recorded/Downloads/ZAM_Over-1_1.xml'
    path9  = "recorded/Custom/2_agents.xml"
    path10 = "recorded/Custom/EMPTY_MAP.xml"
    pathR =  "recorded/hand-crafted/ZAM_Tjunction-1_55_T-1.xml"
    parser.add_argument("--scenario", default=path9)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    if "commonroad" in args.scenario:
        scenario_path = args.scenario.split("scenarios/")[-1]
    else:
        scenario_path = args.scenario

    # load settings from planning_fast.json
    settings_dict = load_planning_json("planning_fast.json")
    settings_dict["risk_dict"] = risk_dict = load_risk_json()
    if not args.time:
        settings_dict["evaluation_settings"]["show_visualization"] = True
    eval_directory = (
        pathlib.Path(__file__).resolve().parents[0].joinpath("results").joinpath("eval")
    )
    # Create the frenet creator
    frenet_creator = FrenetCreator(settings_dict, type_='SAMPLES')

    # Create the scenario evaluator
    evaluator = ScenarioEvaluator(
        planner_creator=frenet_creator,
        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=pathlib.Path(
            os.path.join(mopl_path, "commonroad-scenarios/scenarios/")
        ).resolve(),
        log_path=pathlib.Path("./log/example").resolve(),
        collision_report_path=eval_directory,
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
    )

    def main():
        """Loop for cProfile."""
        _ = evaluator.eval_scenario(scenario_path)

    if args.time:
        import cProfile
        cProfile.run('main()', "output.dat")
        no_trajectores = settings_dict["frenet_settings"]["frenet_parameters"]["n_v_samples"] * len(settings_dict["frenet_settings"]["frenet_parameters"]["d_list"])
        import pstats
        sortby = pstats.SortKey.CUMULATIVE
        with open(f"cProfile/{scenario_path.split('/')[-1]}_{no_trajectores}.txt", "w") as f:
            p = pstats.Stats("output.dat", stream=f).sort_stats(sortby)
            p.sort_stats(sortby).print_stats()
    else:
        main()

# EOF
