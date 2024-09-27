#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import math
import os
from datetime import datetime
from functools import partial
import numpy as np
import casadi as ca
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from typing import List, Dict, Tuple
from std_msgs.msg import Int32
import casadi.tools as ca_tools
from collections import defaultdict
from stl_decomposition_msgs.msg import TaskMsg, LeaderShipTokens, LeafNodes, ImpactMsg
from stl_decomposition_msgs.srv import Impact
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from .dynamics_module import Agent, LeadershipToken, ImpactSolverLP, create_approximate_ball_constraints2d, create_box_constraint_function, create_rectangular_constraint_function
from .builders import (BarrierFunction, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, 
                      create_barrier_from_task, go_to_goal_predicate_2d, formation_predicate, 
                      epsilon_position_closeness_predicate, collision_avoidance_predicate, conjunction_of_barriers)
from tf2_ros import LookupException
from tf_transformations import euler_from_quaternion

class Controller(Node):
    """
    This class is a STL-QP (Signal Temporal Logic-Quadratic Programming) controller for omnidirectional robots and therefore does not consider the orientation of the robots.
    It is responsible for solving the optimization problem and publishing the velocity commands to the agents. 
    """

    def __init__(self):

        # Initialize the node
        super().__init__('controller')
        self.reset_point       : list[int] = [80, 170]          # might be a list in the future if we want more than two sets of tasks
        self.ready_controllers : set[int]  = set()       # Used as a flag to make sure the agents are synked before starting the next set of tasks
        self.current_task_set  : int       = 0
        
        # Velocity Command Message
        self.wheelbase_distance  : float = 0.16
        self.lookahead_distance  : float = 0.12 
        self.max_linear_velocity : float = 0.2
        self.max_angular_velocity: float = 2.0
        self.max_lateral_velocity: float = self.max_angular_velocity*self.lookahead_distance
        # self.agents[self.agent_id].theta               : float = 0.0
        self.vel_cmd_msg  : Twist = Twist()
        self.turtle_msg   : Twist = Twist()

        # Optimization Problem
        self.scale_factor    = 3
        self.initial_time    : float = 0.0 
        self.current_time    : float = 0.0
        self._gamma          : float = 1.0
        self._gamma_tilde    : dict[int,float] = {} 
        self.solver          : ca.Function = None
        self.parameters      : ca_tools.structure3.msymStruct = None
        self.input_vector    = ca.MX.sym('input', 2)
        self.slack_variables : dict[int, ca.MX] = {} 
        self.dummy_scalar    = ca.MX.sym('dummy_scalar', 1)
        self.alpha_fun       = ca.Function('alpha_fun', [self.dummy_scalar], [self.scale_factor * self.dummy_scalar])
        self.barrier_func    : list[ca.Function] = [] 
        self.nabla_funs      : list[ca.Function] = [] 
        self.nabla_inputs    : list[dict[str, ca.MX]] = [] 
        self.enable_collision_avoidance : bool = False     
        self.num_of_planes_for_approx   = 40
        self.A_func, self.b_func = create_rectangular_constraint_function([[-self.max_linear_velocity, self.max_linear_velocity], [-self.max_lateral_velocity, self.max_lateral_velocity]])
        

        # parameters declaration from launch file
        self.declare_parameter('robot_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('num_robots', rclpy.Parameter.Type.INTEGER)
        
        # Agent Information # check if this can be above the optimization problem
        self.total_agents               = self.get_parameter('num_robots').get_parameter_value().integer_value
        self.agent_name                 = self.get_parameter('robot_name').get_parameter_value().string_value
        self.agent_id                   = int(self.agent_name[-1])
        self.latest_self_transform      = TransformStamped()
        self._ready_to_run_service_loop : bool = False 
        self._ready_to_run_control_loop : bool = False 
        self._wait_for_future           : bool = False
        self.agent_positions            : dict[int, list[int,int]] = {} # used to store the position history of the agents
        self.agents                     : dict[int, Agent]  = {}        # position of all the agents in the system including self agent 
        for id in range(1, self.total_agents + 1):
            self.agents[id] = Agent(id=id, state=np.array([0.0, 0.0]), theta=0.0)


        # Information stored about your neighbors
        self.follower_neighbor     : int             = None
        self.leader_neighbors      : set[int]        = set()
        self.leaf_nodes            : list[int]       = []
        self.LeaderShipTokens_dict : dict[tuple,int] = {}
        
        # Impact Information
        self._best_impact_from_leaders   : dict[int,float] = {}     # for each leader you will receive a best impact that you will use to compute your gamma
        self._worst_impact_on_leaders    : dict[int,float] = {}     # after you compute gamma you send back your worst impact to the leaders
        self._worst_impact_from_follower : float = 0.0              # this is the worse impact that the follower of your task will send you back
        self._best_impact_on_follower    : float = 0.0              # this is the best impact that you can have on the task you are leading
        
        # Callback Groups
        self.rc_group        = ReentrantCallbackGroup()
        self.mc_group        = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        # Service and Client Information
        self.ready_to_compute_gamma : bool = False 
        self._impact_backup         : dict[int, float] = {} 
        self.impact_results         : dict[int, float] = {} 
        for id in range(1, self.total_agents + 1): 
            self.impact_results[id] = 0.0
            self._worst_impact_on_leaders[id]  = 0.0
            self._best_impact_from_leaders[id] = 0.0

        # Barriers and tasks
        self.total_tasks                 = float('inf')
        self.task_sets                   : list[list[TaskMsg]]   = [[] for _ in range(len(self.reset_point)+1)] 
        self.task_set_1                  : list[TaskMsg]         = []
        self.task_set_2                  : list[TaskMsg]         = []
        self.barriers                    : list[BarrierFunction] = [] 
        self._barrier_you_are_leading    : BarrierFunction       = None 
        self._barriers_you_are_following : list[BarrierFunction] = [] 
        self._independent_barrier        : BarrierFunction       = None 
        self.task_msg_list               : list[TaskMsg]         = [] 
        

        # Setup publishers
        # self.vel_pub          = self.create_publisher(Twist, f"/agent{self.agent_id}/cmd_vel", 100)
        self.turtle_vel_pub   = self.create_publisher(Twist, f"/turtlebot{self.agent_id}/cmd_vel", 10)
        self.rosie_vel_pub    = self.create_publisher(Twist, f"/rosie0/cmd_vel", 10) 
        self.ready_pub        = self.create_publisher(Int32, "/controller_ready", 10)
        self.agent_pose_pub   = self.create_publisher(PoseStamped, f"/agent{self.agent_id}/agent_pose", 10)
        self.cleared_data_pub = self.create_publisher(Int32, "/cleared_data", 10)

        # Setup subscribers
        self.create_subscription(LeafNodes, "/leaf_nodes", self.leaf_nodes_callback, 10)
        self.create_subscription(Int32, "/numOfTasks", self.numOfTasks_callback, 10)
        self.create_subscription(TaskMsg, "/tasks", self.task_callback, 100)
        self.create_subscription(LeaderShipTokens, "/tokens", self.tokens_callback, 100)
        self.create_subscription(Int32, "/cleared_data", self.cleared_data_callback, 10)
        for id in range(1, self.total_agents + 1):  
            self.create_subscription(PoseStamped, f"/qualisys/turtlebot{id}/pose", 
                                    partial(self.qualisys_pose_callback, agent_id=id), 10, callback_group=self.rc_group)

        # Setup service server
        self.impact_server = self.create_service(Impact, f"/agent{self.agent_id}/impact_service", self.impact_callback, callback_group=self.rc_group)

        # Setup service clients for each other agent
        self.impact_clients = {}
        for id in range(1, self.total_agents + 1):
            if id != self.agent_id:  # Skip creating a client for itself
                client = self.create_client(Impact, f"/agent{id}/impact_service", callback_group=self.client_cb_group)
                self.impact_clients[id] = client
                while not client.wait_for_service(timeout_sec=1.0):
                    self._logger.info(f'Impact service for agent{id} not available, waiting...')


        # Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # This timer is used to update the agent's pose
        # self.check_transform_timer = self.create_timer(0.4, self.transform_timer_callback, callback_group=self.rc_group) 
        
        # This timer is used to wait untill all the tasks are received before it initializes the controller
        self.task_check_timer = self.create_timer(0.8, self.check_tasks_callback, callback_group=self.rc_group) 

        # This timer is used to continuously update the agent's best and/or worst impact
        self.service_timer = self.create_timer(0.4, self.service_loop, callback_group=self.mc_group) 

        # This timer is used to continuously compute the optimized input
        self.control_loop_timer = self.create_timer(0.5, self.control_loop, callback_group=self.mc_group) 

        # This timer is used to reset the controller and reinitialize it
        self.reset_timer = self.create_timer(1.0, self.reset_controller_callback, callback_group=self.mc_group)




    def clear_data(self):

        self.slack_variables = {}
        self.barrier_func    = [] 
        self.nabla_funs      = [] 
        self.nabla_inputs    = [] 
        
        self._ready_to_run_service_loop : bool = False 
        self._ready_to_run_control_loop : bool = False 
        self._wait_for_future           : bool = False 

        self._best_impact_from_leaders   : dict[int,float] = {}     
        self._worst_impact_on_leaders    : dict[int,float] = {}     
        self._worst_impact_from_follower : float = 0.0             
        self._best_impact_on_follower    : float = 0.0             
        
        self._gamma                 : float = 1.0
        self._gamma_tilde           : dict[int,float]  = {}
        self._impact_backup         : dict[int, float] = {} 
        self.impact_results         : dict[int, float] = {} 
        self.ready_controllers      : set[int]         = set()
        self.ready_to_compute_gamma : bool             = False 
        
        for id in range(1, self.total_agents + 1): 
            self.impact_results[id] = 0.0
            self._worst_impact_on_leaders[id]  = 0.0
            self._best_impact_from_leaders[id] = 0.0

        self.barriers                    : list[BarrierFunction] = []
        self.relevant_barriers           : list[BarrierFunction] = [] 
        self._barriers_you_are_following : list[BarrierFunction] = []
        self._barrier_you_are_leading    : BarrierFunction       = None  
        self._independent_barrier        : BarrierFunction       = None 
        

    def reinitialize_controller(self):

        self.clear_data()

        self.barriers = self.create_barriers_from_tasks(self.task_sets[self.current_task_set]) 
        self.solver = self.get_qpsolver()
        self.wait_for_controller_timer = self.create_timer(0.4, self.wait_for_controller, callback_group=self.mc_group)


    def wait_for_controller(self):

        if len(self.ready_controllers) == self.total_agents: 
            self.wait_for_controller_timer.cancel()
            self._ready_to_run_service_loop = True
            self._ready_to_run_control_loop = True
        else:
            agent = Int32()
            agent.data = self.agent_id
            self.cleared_data_pub.publish(agent)
            # pass


    def conjunction_on_same_edge(self, barriers:List[BarrierFunction]) -> List[BarrierFunction]:
        """
        Loop through the barriers and create the conjunction of the barriers on the same edge.

        Args:
            barriers (List[BarrierFunction]): List of all the created barriers.

        Returns:
            new_barriers (List[BarrierFunction]): List of the new barriers created by the conjunction of the barriers on the same edge.
        """
        barriers_dict: defaultdict = defaultdict(list)

        # create the conjunction of the barriers on the same edge
        for barrier in barriers:
            edge = tuple(sorted(barrier.contributing_agents))
            barriers_dict[edge].append(barrier)

        # Use list comprehension to create new_barriers
        new_barriers = [conjunction_of_barriers(barrier_list=barrier_list, associated_alpha_function=self.alpha_fun) 
                        if len(barrier_list) > 1 else barrier_list[0] 
                        for barrier_list in barriers_dict.values()]

        return new_barriers

    
    def _get_splitted_barriers(self,barriers:List[BarrierFunction]) -> Tuple[BarrierFunction,List[BarrierFunction],BarrierFunction]: 
        """This function takes a list of barriers and splits them into two lists. One list contains the barriers that are along the edge where the agent is a leader and the other list contains the barriers that are along the edge where the agent is a follower"""
        
        barrier_you_are_leading   = None
        follower_barriers         = []
        independent_barrier       = None
        
        
        for barrier in barriers:
            involved_agents = barrier.contributing_agents
            
            if len(involved_agents) == 1:
                independent_barrier = barrier
            
            if len(involved_agents) > 1:
                if involved_agents[0] == self.agent_id:
                    neighbor_id = involved_agents[1]
                else:
                    neighbor_id = involved_agents[0]

                normalized_edge = tuple(sorted([self.agent_id, neighbor_id]))

                if self.agent_id == self.LeaderShipTokens_dict[normalized_edge]: 
                    barrier_you_are_leading = barrier
                else:
                    follower_barriers.append(barrier)

        return barrier_you_are_leading, follower_barriers, independent_barrier
    

    def get_leader_and_follower_neighbors(self):
        for edge, token in self.LeaderShipTokens_dict.items():

            if self.agent_id in edge:

                if edge[0] == self.agent_id:
                    neighbor_id = int(edge[1])
                else :
                    neighbor_id = int(edge[0])
            
                if token == self.agent_id: # check if you are the leader of the tasks on this edge
                    self.follower_neighbor = neighbor_id
                else:
                    self.leader_neighbors.add(neighbor_id)


    def create_barriers_from_tasks(self, messages:List[TaskMsg]) -> List[BarrierFunction]:
        """
        Constructs the barriers from the subscribed task messages.     
        
        Args:
            messages (List[TaskMsg]): List of task messages.
        
        Returns:
            barriers_list (List[BarrierFunction]): List of the created barriers.
        
        Raises:
            Exception: If the task type is not supported.  

        Note:
            The form of the task message is defined in the custom_msg package and looks like this:
            int32[] edge
            string type
            float32[] center
            float32 epsilon
            string temp_op
            int32[] interval
            int32[] involved_agents 
            float32 start
        """
        barrier_list = []

        for message in messages:

            # Create the predicate based on the type of the task
            if message.type == "go_to_goal_predicate_2d":
                predicate = go_to_goal_predicate_2d(goal=np.array(message.center), epsilon=message.epsilon, 
                                                    agent=self.agents[message.involved_agents[0]])
            elif message.type == "formation_predicate":
                predicate = formation_predicate(epsilon=message.epsilon, agent_i=self.agents[message.involved_agents[0]], 
                                                agent_j=self.agents[message.involved_agents[1]], relative_pos=np.array(message.center))
            elif message.type == "epsilon_position_closeness_predicate":
                predicate = epsilon_position_closeness_predicate(epsilon=message.epsilon, agent_i=self.agents[message.involved_agents[0]], 
                                                                 agent_j=self.agents[message.involved_agents[1]])
            else:
                raise Exception(f"Task type {message.type} is not supported")
            
            # Create the temporal operator
            if message.temp_op == "AlwaysOperator":
                temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=message.interval[0], b=message.interval[1]))
            elif message.temp_op == "EventuallyOperator":
                temporal_operator = EventuallyOperator(time_interval=TimeInterval(a=message.interval[0], b=message.interval[1]))

            # Create the task
            task = StlTask(predicate=predicate, temporal_operator=temporal_operator)

            # Add the task to the barriers and the edge
            initial_conditions = [self.agents[i] for i in message.involved_agents] # Needs the current state of the agents after the reset
            # for agent in initial_conditions:
            #     self._logger.info(f"Agent {agent.id} state: {agent.state}")

            # Get the current time
            time_in_sec,_ = self.get_clock().now().seconds_nanoseconds()
            self.current_time = ca.vertcat(time_in_sec - self.initial_time)
            
            barrier_list += [create_barrier_from_task(task=task, initial_conditions=initial_conditions, alpha_function=self.alpha_fun, t_init=self.current_time)]
            barrier_list = self.conjunction_on_same_edge(barrier_list)

        return barrier_list


    def controller_parameters(self) -> ca_tools.structure3.msymStruct:
        """
        Initializes the parameter structure that contains the state of the agents and the time. 
        
        Returns:
            parameters (ca_tools.structure3.msymStruct): The parameter structure.
        """
        # Create the parameter structure for the optimization problem
        parameter_list  = []
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in self.agents.keys()]
        parameter_list += [ca_tools.entry("time", shape=1)]
        parameter_list += [ca_tools.entry("gamma", shape=1)]
        parameter_list += [ca_tools.entry("theta", shape=1)]

        
        if self.follower_neighbor is not None: # if the agent does not have a follower then it does not need to compute the best impact for the follower and won't get the worst impact from the follower
            parameter_list += [ca_tools.entry("epsilon", shape=1)]
        
        if self.enable_collision_avoidance:
            parameter_list +=  [ca_tools.entry('collision_pos_'+str(id),shape=2) for id in range(1, self.total_agents + 1) if id != self.agent_id]  # one parameter for the state of any obstacle met by the agent
            parameter_list +=  [ca_tools.entry('collision_switch_'+str(id),shape=1)  for id in range(1, self.total_agents + 1) if id != self.agent_id]  # used to switch off a collision avoidance constraint when not needed
            parameter_list +=  [ca_tools.entry('collision_load_'+str(id),shape=1)  for id in range(1, self.total_agents + 1) if id != self.agent_id]  # used to switch off a collision avoidance constraint when not needed
            
        parameters = ca_tools.struct_symMX(parameter_list)

        return parameters


    def generate_barrier_constraints(self, barrier_list:List[BarrierFunction]) -> ca.MX:
        """
        Iterates through the barrier list and generates the constraints for each barrier by calculating the gradient of the barrier function.
        It also creates the slack variables for the constraints.

        Args:
            barrier_list (List[BarrierFunction]): List of the conjuncted barriers.

        Returns:
            constraints (ca.MX): The constraints for the optimization problem.
        """
        constraints = []
        for barrier in barrier_list:
            # Check the barrier for leading agent
            if len(barrier.contributing_agents) > 1:
                if barrier.contributing_agents[0] == self.agent_id:
                    neighbor_id = barrier.contributing_agents[1]
                else:
                    neighbor_id = barrier.contributing_agents[0]
            else :
                neighbor_id = self.agent_id
            
            # Create the named inputs for the barrier function
            named_inputs = {"state_"+str(id): self.parameters["state_"+str(id)] for id in barrier.contributing_agents}
            named_inputs["time"] = self.parameters["time"]

            # Get the necessary functions from the barrier
            nabla_xi_fun                = barrier.gradient_function_wrt_state_of_agent(self.agent_id)
            barrier_fun                 = barrier.function
            partial_time_derivative_fun = barrier.partial_time_derivative

            # Calculate the symbolic expressions for the barrier constraint
            nabla_xi = nabla_xi_fun.call(named_inputs)["value"]
            dbdt     = partial_time_derivative_fun.call(named_inputs)["value"]
            alpha_b  = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"])

            # if it is a self task
            if neighbor_id == self.agent_id:
                load_sharing = 1
                barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b)) 
            else:
                # check edge for leading agent
                leader = self.LeaderShipTokens_dict[tuple(sorted([self.agent_id, neighbor_id]))]         

                if leader == self.agent_id:
                    load_sharing = 1
                    barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b) + self.parameters['epsilon']) 
                elif leader == neighbor_id:
                    load_sharing = 0.5  
                    slack = ca.MX.sym(f"slack", 1)
                    self.slack_variables[neighbor_id] = slack
                    barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b) + slack) 

            constraints.append(barrier_constraint)
            self.barrier_func.append(barrier_fun)
            self.nabla_funs.append(nabla_xi_fun)
            self.nabla_inputs.append(named_inputs)

        return ca.vertcat(*constraints)


    def _get_collision_avoidance_barrier(self)->None:

        x  = ca.MX.sym("x",2)           # state of the agent (which also constains the position)
        y  = ca.MX.sym("y",2)           # position of the obstacle
        switch = ca.MX.sym("switch",1)  # switch off the constraint when not needed
        load   = ca.MX.sym("load",1)    # switch off the constraint when not needed

        collision_radius = 0.20                              # assuming the two agents are 16 cm big
        barrier = (x-y).T@(x-y) - (2*collision_radius)**2    # here the collsion radius is assumed to be 36 cm for each object 

        g_xu = np.eye(self.agents[self.agent_id].state.size)@self.input_vector
        db_dx = ca.jacobian(barrier,x)

        # Evaluate alpha function with the barrier
        alpha_barrier = 0.5*barrier                        
        constraint = db_dx @ g_xu + load * (alpha_barrier) # if load = 0.5 -> cooperative collsion. If load =1 , then non cooperative
        
        #(-1) factor needed to turn the constraint into negative g(x)<=0
        # switch -> it will be 1 if it is needed the constraint and 0 if not
        collision_constraint_fun = ca.Function("collision_avoidance",[x,y,switch,load,self.input_vector],
                                          [-1*(constraint)*switch]) # create a function that will be used to compute the collision avoidance constraints
        
        return collision_constraint_fun

 
    def _get_collision_avoidance_constraints(self,parameters) -> ca.MX:
        """ Here we create the collision avoidance solver """
        
        collision_contraints = []
        for id in range(1, self.total_agents + 1): # Reduce the number of constraints by making the agent only avoid its leader_neighbors
            if id != self.agent_id:
                collision_contraints += [self._collision_constraint_fun(parameters["state_"+str(self.agent_id)],
                                                                        parameters["collision_pos_"+str(id)],
                                                                        parameters["collision_switch_"+str(id)],
                                                                        parameters["collision_load_"+str(id)],
                                                                        self.input_vector)]

        return ca.vertcat(*collision_contraints)


    def get_input_constraints(self) -> ca.MX:
        """
        This function creates the input constraints for the optimization problem.
        
        """
        A_func = self.A_func
        b_func = self.b_func

        A = A_func(self.parameters["theta"])
        b = b_func(self.parameters["theta"])

        input_constraints = A @ self.input_vector - self.parameters["gamma"] * b

        return input_constraints


    def get_qpsolver(self) -> ca.qpsol:
        """
        This function creates all that is necessary for the optimization problem and creates the optimization solver.

        Returns:
            solver (ca.qpsol): The optimization solver.

        Note:
            The cost function is a quadratic function of the input vector and the slack variables where the slack variables are used to enforce the barrier constraints.
        
        """

        

        
        # get the neighbors of the current agent
        self.get_leader_and_follower_neighbors()
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        self._barrier_you_are_leading, self._barriers_you_are_following, self._independent_barrier  = self._get_splitted_barriers(barriers = self.relevant_barriers)

        # Create the parameter structure for the optimization problem --- 'p' ---
        self.parameters = self.controller_parameters()
        # Create the impact solver that will be used to compute the best and worst impacts on the barriers
        self._impact_solver : ImpactSolverLP = ImpactSolverLP(agent=self.agents[self.agent_id], max_velocity=[self.max_linear_velocity,self.max_lateral_velocity])


        # Create the constraints for the optimization problem --- 'g' ---
        # input_constraints = self.A @ self.input_vector - self.parameters["gamma"] * self.b
        input_constraints      = self.get_input_constraints() 
        barrier_constraints    = self.generate_barrier_constraints(self.relevant_barriers)
        slack_constraints      = - ca.vertcat(*list(self.slack_variables.values()))
        
        if self.enable_collision_avoidance:
            self._collision_constraint_fun = self._get_collision_avoidance_barrier()
            collision_constraints          = self._get_collision_avoidance_constraints(self.parameters) 
            constraints                    = ca.vertcat(input_constraints, barrier_constraints, slack_constraints, collision_constraints)
        else:
            constraints                    = ca.vertcat(input_constraints, barrier_constraints, slack_constraints)

        # Create the decision variables for the optimization problem --- 'x' ---
        slack_vector = ca.vertcat(*list(self.slack_variables.values()))
        opt_vector   = ca.vertcat(self.input_vector, slack_vector)

        # Create the object function for the optimization problem --- 'f' ---
        Q = ca.diag(ca.vertcat(1,1))
        R = ca.vertcat(ca.horzcat(ca.cos(self.parameters["theta"]), -ca.sin(self.parameters["theta"])),
                       ca.horzcat(ca.sin(self.parameters["theta"]), ca.cos(self.parameters["theta"])))
        
        cost = self.input_vector.T @ R @ Q @ R.T @ self.input_vector
        for id,slack in self.slack_variables.items():
            if id == self.agent_id:
                cost += 100* slack**2 
            else:
                cost += 10* slack**2

        # Create the optimization solver
        qp = {'x': opt_vector, 'f': cost, 'g': constraints, 'p': self.parameters}
        solver = ca.qpsol('sol', 'qpoases', qp, {'printLevel': 'none'})


        return solver




    def control_loop(self):
        """This is the main control loop of the controller. It calculates the optimal input and publishes the velocity command to the cmd_vel topic."""
        
        if self._ready_to_run_control_loop:
            
            # Get the current time
            time_in_sec = self.get_clock().now().nanoseconds/1e9
            self.current_time = ca.vertcat(time_in_sec - self.initial_time)

            # Fill the structure with the values
            current_parameters = self.parameters(0)

            current_parameters["time"]  = self.current_time
            current_parameters["gamma"] = self._gamma
            current_parameters["theta"] = self.agents[self.agent_id].theta#self.agents[self.agent_id].theta

            for id in self.agents.keys():
                current_parameters[f'state_{id}'] = self.agents[id].state#ca.vertcat(self.agents[id].state[0], self.agents[id].state[1])

            if self.follower_neighbor is not None:
                current_parameters["epsilon"] = self._worst_impact_from_follower
                # self._logger.info(f"epsilon from agent {self.follower_neighbor}: {self._worst_impact_from_follower}")

            if self.enable_collision_avoidance:
                current_agent_pos = ca.vertcat(self.agents[self.agent_id].state[0], self.agents[self.agent_id].state[1])
                # current_agent_pos = self.agents[self.agent_id].state
                
                for id in range(1, self.total_agents + 1):
                    if id != self.agent_id:
                        # other_agent_pos = self.agents[id].state
                        other_agent_pos = ca.vertcat(self.agents[id].state[0], self.agents[id].state[1])
                        distance = ca.norm_2(current_agent_pos - other_agent_pos)
                        # self._logger.info(f"Agent {self.agent_id}:{current_agent_pos} and agent {id}:{other_agent_pos} is {distance} apart")
                    
                        current_parameters["collision_pos_" + str(id)] = other_agent_pos
                        
                        if distance < 1.5: # 2.0
                            current_parameters["collision_switch_" + str(id)] = 1.0
                            current_parameters["collision_load_" + str(id)] = 1.0
                        else:
                            current_parameters["collision_switch_" + str(id)] = 0.0
                            current_parameters["collision_load_" + str(id)] = 0.0 
                        

                        
            # Calculate the gradient values to check for convergence
            inputs = {}
            nabla_list = []

            for i, nabla_fun in enumerate(self.nabla_funs):
                inputs = {key: current_parameters[key] for key in self.nabla_inputs[i].keys()}
                # self._logger.info(f"current time: {current_parameters['time']}")
                # self._logger.info(f"barrier {self.barrier_func[i].name_in()} : {self.barrier_func[i].call(inputs)['value']}")
                nabla_val = nabla_fun.call(inputs)["value"]
                nabla_list.append(ca.norm_2(nabla_val))

            # self._logger.info(f"Norm of the gradient values: {nabla_list}")


            # Solve the optimization problem 
            if any(ca.norm_2(val) < 1e-10 for val in nabla_list):
                optimal_input = ca.MX.zeros(2 + len(self.slack_variables))
            else:
                sol = self.solver(p=current_parameters, ubg=0)
                optimal_input = sol['x']
            
            if self.agent_id == 7:
                # Publish the velocity command
                linear_velocity = optimal_input[:2]
                clipped_linear_velocity = np.clip(linear_velocity, -self.max_linear_velocity, self.max_linear_velocity)
                self.vel_cmd_msg.linear.x = clipped_linear_velocity[0][0]
                self.vel_cmd_msg.linear.y = clipped_linear_velocity[1][0]
            
                self.rosie_vel_pub.publish(self.vel_cmd_msg)
            else:

                # Define the clockwise rotation matrix R(theta)
                theta_val = current_parameters["theta"] 
                R = ca.vertcat(ca.horzcat(ca.cos(theta_val), -ca.sin(theta_val)),
                            ca.horzcat(ca.sin(theta_val), ca.cos(theta_val)))

                # Extract the optimal input velocity vector and calculate the rotated velocity vector 
                u         = optimal_input[:2]         # u = [Vx, Vy]^T
                u_rotated = ca.mtimes(R.T, u)           # u_perpend = [v, w*L]^T

                # Extract linear and angular velocities from the rotated vector (THESE ARE QUATERION VALUES)
                linear_velocity  = u_rotated[0]                             # v (linear velocity)
                angular_velocity = u_rotated[1] / self.lookahead_distance   # w (angular velocity)
                
                clipped_linear_velocity = np.clip(linear_velocity, -self.max_linear_velocity, self.max_linear_velocity)
                clipped_angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

                self.turtle_msg.linear.x = clipped_linear_velocity[0][0]
                self.turtle_msg.angular.z = clipped_angular_velocity[0][0]

                if self.agent_id == 1:
                    # self._logger.info("==============================")
                    self._logger.info(f"Current time: {self.current_time}")
                    # self._logger.info(f"Theta controller: {theta_val}")  
                    # self._logger.info(f"Vx: {optimal_input[0]}, Vy: {optimal_input[1]}")
                    # self._logger.info(f"v: {linear_velocity}, w: {angular_velocity}")
                    # self._logger.info(f"linear velocity: {self.turtle_msg.linear.x:.3f}, angular velocity: {self.turtle_msg.angular.z:.3f}")
                    # self._logger.info("==============================")

                self.turtle_vel_pub.publish(self.turtle_msg)

            # if self.agent_id == 7:
            #     self.rosie_vel_pub.publish(self.turtle_msg)
            # else:
            #     self.turtle_vel_pub.publish(self.turtle_msg)

            # reset values for the next iteration
            self._gamma_tilde = {}
            self._gamma = 1.0
            # self._logger.info(f"Current time: {self.current_time}")

        else:
            pass
        


    def service_loop(self): 
        
        if self._ready_to_run_service_loop:

            if self.agent_id in self.leaf_nodes:
                self.process_leaf_node()
            else:
                self.process_non_leaf_node()
        else:
            pass
        
            
    def process_leaf_node(self):

        # self._logger.info("computing gamma and best impact")
        self.compute_gamma_and_best_impact_on_leading_task()
        # self._logger.info(f"sending a request to the follower {self.follower_neighbor} for the worst impact")
        response = self.call_impact_callback(self.follower_neighbor, "worst")
        if response is not None:
            # self._logger.info(f"I now have the worst impact from follower {self.follower_neighbor}")
            self._worst_impact_from_follower = response


    def process_non_leaf_node(self):
        for leader in self.leader_neighbors:
            # self._logger.info(f"sending a request to the leader {leader} for the best impact")
            response = self.call_impact_callback(leader, "best")
            if response is not None:
                # self._logger.info(f"I now have the best impact from leader {leader}")
                self._best_impact_from_leaders[leader] = response

        # self._logger.info("all best impacts are received")
        self.compute_gamma_tilde_values()

        if self.ready_to_compute_gamma:
            # self._logger.info("computing gamma and best impact")
            self.compute_gamma_and_best_impact_on_leading_task()
            self.ready_to_compute_gamma = False
            # self._logger.info("computing worst impact")
            self.compute_worst_impact_on_following_task()
        else:
            self._logger.info("Have not computed all gamma tilde values yet. Retry later...")

        if self.follower_neighbor is not None:
            # self._logger.info(f"sending a request to the follower {self.follower_neighbor} for the worst impact")
            response = self.call_impact_callback(self.follower_neighbor, "worst")
            if response is not None:
                # self._logger.info(f"I now have the worst impact from follower {self.follower_neighbor}")
                self._worst_impact_from_follower = response




    def compute_gamma_tilde_values(self):

        # self._logger.info(f"Trying to compute gamma tilde values...")
        for barrier in self._barriers_you_are_following:
            involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
            
            if involved_agent[0] == self.agent_id:
                neighbor_id = involved_agent[1]
            else :
                neighbor_id = involved_agent[0]
            
            if not (neighbor_id in self._gamma_tilde.keys()):
                gamma_tilde = self._compute_gamma_for_barrier(barrier)
                if gamma_tilde != None:
                    self._gamma_tilde[neighbor_id] = gamma_tilde
        # self._logger.info(f"gamma tilde values for agent {self.agent_id}: {self._gamma_tilde}")
        if len(self._gamma_tilde) == len(self.leader_neighbors):
            self.ready_to_compute_gamma = True
        else:
            self._logger.error(f"Only {len(self._gamma_tilde)} out of {len(self.leader_neighbors)} gamma tilde values are computed.")
        
          
    def _compute_gamma_for_barrier(self, barrier: BarrierFunction) -> float :
    
        involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
        
        if len(involved_agent) ==1:
            raise RuntimeError("The given barrier function is a self task. Gamma should not be computed for this task. please revise the logic of the contoller")
        else :
             
            if involved_agent[0] == self.agent_id:
                neighbor_id = involved_agent[1]
            else :
                neighbor_id = involved_agent[0]
                
        
        barrier_fun    : ca.Function   = barrier.function
        local_gradient : ca.Function   = barrier.gradient_function_wrt_state_of_agent(self.agent_id)    
        
        associated_alpha_function :ca.Function  = barrier.associated_alpha_function # this is the alpha function associated to the barrier function in the barrier constraint
        partial_time_derivative:ca.Function     = barrier.partial_time_derivative
        
        if associated_alpha_function == None:
            raise RuntimeError("The alpha function associated to the barrier function is null. please remeber to store this function in the barrier function object for barrier computation")

        neigbour_state       = self.agents[neighbor_id].state    # the neighbor state
        current_agent_state  = self.agents[self.agent_id].state  # your current state
        time                 = self.current_time
        named_inputs         = {"state_"+str(self.agent_id):current_agent_state.flatten(),"state_"+str(neighbor_id):neigbour_state.flatten(),"time":time}

        local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient
        
        g_value = np.eye(current_agent_state.size)

        if local_gardient_value.shape[0] == 1:
            local_gardient_value = local_gardient_value.T

        if neighbor_id in self.leader_neighbors: 
            # this can be parallelised for each of the barrier you are a follower of
            worst_input = self._impact_solver.minimize(Lg = local_gardient_value.T @ g_value, theta=np.array([self.agents[self.agent_id].theta])) # find the input that minimises the dot product with Lg given the bound on the input
            if worst_input.shape[0] == 1:
                worst_input = worst_input.T
        
            # now we need to compute the gamma for this special case  
            try :
                neighbor_best_impact = self._best_impact_from_leaders[neighbor_id] # this is a scalar value representing the best impact of the neigbour on the barrier given its intupt limitations
                # self._logger.info(f"Upack leader best impact from agent {neighbor_id} with value {neighbor_best_impact}") 
            except:
                # self._logger.info(f"Required leaders best impact from agent {neighbor_id} not available yet. Retry later...") 
                return None
            
            alpha_barrier_value            = associated_alpha_function(barrier_fun.call(named_inputs)["value"])                # compute the alpha function associated to the barrier function
            partial_time_derivative_value  = partial_time_derivative.call(named_inputs)["value"]                               # compute the partial time derivative of the barrier function
            
            zeta = alpha_barrier_value + partial_time_derivative_value 
            
            # self._logger.info(f"Alpha barrier value: {alpha_barrier_value}")
            # self._logger.info(f"Partial time derivative value: {partial_time_derivative_value}")
            # self._logger.info(f"Zeta value: {zeta}")

            if np.linalg.norm(local_gardient_value) <= 10**-6 :
                gamma_tilde = 1
            else : #TODO: Introduce a reduction factor for collision avoidance
                gamma_tilde =  -(neighbor_best_impact + zeta) / ( local_gardient_value.T @ g_value @ worst_input) # compute the gamma value
                
                if alpha_barrier_value < 0 :
                    self._logger.warning(f"Alpha barrier value is negative. This entails task dissatisfaction. The value is {alpha_barrier_value}. Please verify that the task is feasible")
                return float(gamma_tilde)
   
    
    def compute_gamma_and_best_impact_on_leading_task(self) : 

        # now it is the time to check if you have all the available information
        if len(self.leader_neighbors) == 0 : 
            self._gamma = 1
        elif len(self._gamma_tilde) != len(self.leader_neighbors) :
            raise RuntimeError(f"The list of gamma tilde values is not complete. You have {len(self._gamma_tilde)} values, but you should have {len(self.leader_neighbors)} values from each of the leaders. Make sure you compute the gamma_tilde value for each leader at this iteration.")
        else:
            gamma_tildes_list = list(self._gamma_tilde.values())
            self._gamma = min(gamma_tildes_list + [1]) # take the minimum of the gamma tilde values
            
            if self._gamma<=0 :
                self._logger.error(f"The computed gamma value is negative. This breakes the process. The gamma value is {self._gamma}")
                raise RuntimeError(f"The computed gamma value for agent {self.agent_id} is negative. This breakes the process. The gamma value is {self._gamma}")
            
        
        # now that computation of the best impact is undertaken
        if self.follower_neighbor is not None : # if you have a task you are leader of then you should compute your best impact for the follower agent
            # now compute your best input for the follower agent

            local_gradient :ca.Function = self._barrier_you_are_leading.gradient_function_wrt_state_of_agent(self.agent_id)    

            named_inputs   = {"state_"+str(self.agent_id)         :self.agents[self.agent_id].state,
                            "state_"+str(self.follower_neighbor):self.agents[self.follower_neighbor].state,
                            "time"                                 :self.current_time}

            local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient
            g_value = np.eye(self.agents[self.agent_id].state.size)

            # then you are leader of this task and you need to compute your best impact
            best_input = self._impact_solver.maximize(Lg= local_gardient_value@g_value, theta=np.array([self.agents[self.agent_id].theta])) # sign changed because you want the maximum in reality. 
            
            if best_input.shape[0] == 1:
                best_input = best_input.T
            if local_gardient_value.shape[0] == 1:
                local_gardient_value = local_gardient_value.T

            
            best_impact_value = np.dot(local_gardient_value.T,(g_value @ best_input*self._gamma)) # compute the best impact of the leader on the barrier given the gamma_i
            best_impact_value = np.squeeze(best_impact_value)
            self._best_impact_on_follower = float(best_impact_value)

  
    def compute_worst_impact_on_following_task(self) :
        
        # if you have leaders to notify then do it
        if len(self.leader_neighbors) != 0 :
            
            for barrier in self._barriers_you_are_following:
                # now compute your best input for the follower agent
                involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
                
                if len(involved_agent) > 1:
                    if involved_agent[0] == self.agent_id:
                        leader_neighbor = involved_agent[1]
                    else :
                        leader_neighbor = involved_agent[0]
                else : # single agent task doesn't need any worst_input computation
                    continue
                    
                
                local_gradient :ca.Function = barrier.gradient_function_wrt_state_of_agent(self.agent_id)    
                
                named_inputs   = {"state_"+str(self.agent_id)        :self.agents[self.agent_id].state,
                                  "state_"+str(leader_neighbor)      :self.agents[leader_neighbor].state,
                                  "time"                              :self.current_time}

                local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient

                g_value = np.eye(self.agents[self.agent_id].state.size)

                # then you are leader of this task and you need to compute your best impact
                worst_input = self._impact_solver.minimize(Lg= local_gardient_value@g_value, theta=np.array([self.agents[self.agent_id].theta]))
                
                
                if worst_input.shape[0] == 1:
                    worst_input = worst_input.T
                if local_gardient_value.shape[0] == 1:
                    local_gardient_value = local_gardient_value.T

                worst_impact_value = np.dot(local_gardient_value.T,(g_value @ worst_input*self._gamma))
                worst_impact_value = np.squeeze(worst_impact_value)
                self._worst_impact_on_leaders[leader_neighbor] = float(worst_impact_value)






    #  ==================== Callbacks ====================



        


    def qualisys_pose_callback(self, msg: PoseStamped, agent_id):
        """
        Callback function to send the qualisys pose of the agent to the other agents.
        
        Args:
            msg (PoseStamped): The pose message of the agent.
        """
        (roll, pitch, yaw) = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        self.agents[agent_id].state = np.array([msg.pose.position.x, msg.pose.position.y])
        self.agents[agent_id].theta = yaw
        

        # Store position in agent's position history
        if agent_id not in self.agent_positions:
            self.agent_positions[agent_id] = []  # Initialize position history for this agent
        
        # Append current position to position history
        self.agent_positions[agent_id].append([msg.pose.position.x, msg.pose.position.y])

        # Signals the manager node that it is ready to recive the tasks 
        ready = Int32()
        ready.data = agent_id
        self.ready_pub.publish(ready)



    def leaf_nodes_callback(self, msg):
            """
            Callback function to get the leaf nodes in the communication graph.

            Args:
                msg (LeafNodes): The leaf nodes message.
            """
            self.leaf_nodes = msg.list


    def numOfTasks_callback(self, msg):
        """
        Callback function for the total number of tasks and is used as a flag to wait for all tasks to be received.
        
        Args:
            msg (Int32): The total number of tasks.
        """
        self.total_tasks = msg.data


    def task_callback(self, msg):
        """
        Callback function for the task messages.
        
        Args:
            msg (TaskMsg): The task message.
        """
        self.task_msg_list.append(msg)


    def check_tasks_callback(self): 
        """Check if all tasks have been received."""

        if len(self.task_msg_list) == self.total_tasks: 

            self.task_check_timer.cancel()  # Stop the timer if all tasks are received
            self._logger.info("==All tasks received==")
            self.initial_time,_ = self.get_clock().now().seconds_nanoseconds()
            
            # Create the tasks and the barriers
            for task in self.task_msg_list:
                assigned = False
                for i, reset_point in enumerate(self.reset_point):
                    if task.interval[1] <= reset_point:
                        self.task_sets[i].append(task)
                        assigned = True
                        break
                if not assigned:
                    self.task_sets[-1].append(task)


            self.barriers = self.create_barriers_from_tasks(self.task_sets[0])
            self.solver = self.get_qpsolver()

            self._ready_to_run_service_loop = True
            self._ready_to_run_control_loop = True

        else:
            # self._logger.info(f"Waiting for {self.total_tasks - len(self.task_msg_list)} tasks to be received...")
            pass

    
    def reset_controller_callback(self):
        """Check if it is time to reset the controller."""
        
        if self.current_time > self.reset_point[self.current_task_set]:

            if self.current_task_set == len(self.reset_point) - 1:
                self.reset_timer.cancel()
            
            self.current_task_set += 1
            self._ready_to_run_service_loop = False 
            self._ready_to_run_control_loop = False 

            self.reinitialize_controller()
            
        else:
            pass

    def cleared_data_callback(self, msg):
        agent = msg.data
        self.ready_controllers.add(agent)



    def tokens_callback(self, msg):
        """
        Callback function for the tokens message.
        
        Args:
            msg (LeaderShipTokens): The tokens message.
        """
        tokens_list = msg.list
        for edgeleadership in tokens_list:
            node1, node2, leader = edgeleadership.i, edgeleadership.j, edgeleadership.leader

            normalized_edge = tuple(sorted([node1, node2]))
            if normalized_edge not in self.LeaderShipTokens_dict:
                self.LeaderShipTokens_dict[normalized_edge] = leader


    def call_impact_callback(self, neighbor_id: int, impact_type: str):
        """
        Callback funtion to construct the request message and send it to the impact service.

        Args:
            neighbor_id (int): The ID of the neighbor agent to send the request to.
            impact_type (str): The type of the impact (best or worst) you want.
        """

        request = Impact.Request()
        request.i = self.agent_id
        request.j = neighbor_id
        request.type = impact_type

        self.future = self.impact_clients[neighbor_id].call_async(request)
        self.future.add_done_callback(self.future_callback)
        
        while not self._wait_for_future:
            pass

        self._wait_for_future = False

        if self.future.done() and self.future.result() is not None:
            # self._logger.info(f"Future is done and received response from agent {neighbor_id} with impact value {self.future.result().impact}")
            return self.future.result().impact
        else:
            self._logger.error(f'Service call failed or timed out, sending backup value {float(self._impact_backup.get(neighbor_id, 0.0))}')
            return float(self._impact_backup.get(neighbor_id, 0.0))


    def impact_callback(self, request, response):
        """ 
        Callback function for the impact service. 
        Agent will ecive a request fron another agent and it will send back the response with the best or worst impact value.
        
        Args: 
            request (ImpactRequest): The request message.
            response (ImpactResponse): The response message.
        """

        # self._logger.info(f"Received request from agent {request.i} for {request.type} impact")
        if request.type == "best":
            if self._best_impact_on_follower != 0.0:
                response.impact = float(self._best_impact_on_follower)
                self._impact_backup[request.i] = response.impact
                self._best_impact_on_follower = 0.0
                # self._logger.info(f"sending response to agent {request.i} with best impact value {response.impact}")
            else:
                # self._logger.info(f"best impact not computed yet, using backup value: {self._impact_backup.get(request.i, 0.0)}")
                response.impact = float(self._impact_backup.get(request.i, 0.0))

        elif request.type == "worst":
            if self._worst_impact_on_leaders[request.i] != 0.0: 
                response.impact = float(self._worst_impact_on_leaders[request.i])
                self._impact_backup[request.i] = response.impact
                self._worst_impact_on_leaders[request.i] = 0.0 
                # self._logger.info(f"sending response to agent {request.i} with worst impact value {response.impact}")
            else:
                # self._logger.info(f"worst impact not computed yet, using backup value: {self._impact_backup.get(request.i, 0.0)}")
                response.impact = float(self._impact_backup.get(request.i, 0.0))

        return response
    

    def future_callback(self, future):
        """ Callback function for the future object. """
        self._wait_for_future = True


    def save_positions_to_file(self):
        """Method to save agent positions to a text file."""
        if self.agent_id == 1:

            for agent_id, positions in self.agent_positions.items():
                with open(f'agent_{agent_id}_positions.txt', 'w') as file:
                    for position in positions:
                        file.write(f"{position[0]} {position[1]}\n")    

    # def save_positions_to_file(self):
    #     """Method to save agent positions to a text file, creating a new folder each time."""
    #     if self.agent_id == 1:
    #         # Create a new folder with a timestamp to ensure uniqueness
    #         folder_name = datetime.now().strftime("positions_%Y%m%d_%H%M%S")
    #         os.makedirs(folder_name, exist_ok=True)

    #         # Loop through each agent's positions and save them in the new folder
    #         for agent_id, positions in self.agent_positions.items():
    #             file_path = os.path.join(folder_name, f'agent_{agent_id}_positions.txt')
    #             with open(file_path, 'w') as file:
    #                 for position in positions:
    #                     file.write(f"{position[0]} {position[1]}\n")

    

def main(args=None):
    rclpy.init(args=args)
 
    node = Controller()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.save_positions_to_file()
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()