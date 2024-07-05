#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import time
from functools import partial
import numpy as np
import casadi as ca
from rclpy.executors import MultiThreadedExecutor
from typing import List, Dict, Tuple
from std_msgs.msg import Int32
import casadi.tools as ca_tools
from collections import defaultdict
from stl_decomposition_msgs.msg import TaskMsg, LeaderShipTokens, LeafNodes, ImpactMsg
from stl_decomposition_msgs.srv import Impact
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from .dynamics_module import Agent, LeadershipToken, ImpactSolverLP, create_approximate_ball_constraints2d
from .builders import (BarrierFunction, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, 
                      create_barrier_from_task, go_to_goal_predicate_2d, formation_predicate, 
                      epsilon_position_closeness_predicate, conjunction_of_barriers)
from tf2_ros import LookupException


class Controller(Node):
    """
    This class is a STL-QP (Signal Temporal Logic-Quadratic Programming) controller for omnidirectional robots and therefore does not consider the orientation of the robots.
    It is responsible for solving the optimization problem and publishing the velocity commands to the agents. 
    """

    def __init__(self):
        # Initialize the node
        super().__init__('controller')

        # Velocity Command Message
        self.max_velocity = 1.0
        self.vel_cmd_msg = Twist()

        # Optimization Problem
        self.solver : ca.Function = None
        self.parameters : ca_tools.structure3.msymStruct = None
        self.input_vector = ca.MX.sym('input', 2)
        self.slack_variables = {}
        self.scale_factor = 3
        self.dummy_scalar = ca.MX.sym('dummy_scalar', 1)
        self.alpha_fun = ca.Function('alpha_fun', [self.dummy_scalar], [self.scale_factor * self.dummy_scalar])
        self.barrier_func = []
        self.nabla_funs = []
        self.nabla_inputs = []
        self.initial_time :float = 0
        self.current_time :float = 0
        self._gamma : float = 1
        self._gamma_tilde :dict[int,float] = {}
        self.A, self.b, self.input_verticies = create_approximate_ball_constraints2d(radius=self.max_velocity, points_number=40)
        
        # parameters declaration from launch file
        self.declare_parameter('robot_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('num_robots', rclpy.Parameter.Type.INTEGER)
        
        # Agent Information
        self.agent_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.agent_id = int(self.agent_name[-1])
        self.latest_self_transform = TransformStamped()
        self.total_agents = self.get_parameter('num_robots').get_parameter_value().integer_value
        self.agents : dict[int, Agent]= {} # position of all the agents in the system including self agent
        
        # Information stored about your neighbours
        self.LeaderShipTokens_dict : Dict[tuple,int]= {}
        self._leadership_tokens : Dict[int,LeadershipToken] = {}
        self.follower_neighbour : int = None
        self.leader_neighbours : list[int] = []
        self.leaf_nodes : list[int] = []
        
        self._best_impact_from_leaders   : dict[int,float] = {}     # for each leader you will receive a best impact that you will use to compute your gamma
        self._worst_impact_on_leaders    : dict[int,float] = {}     # after you compute gamma you send back your worst impact to the leaders
        
        self._worst_impact_from_follower : float = 0.0             # this is the worse impact that the follower of your task will send you back
        self._best_impact_on_follower    : float = 0.0             # this is the best impact that you can have on the task you are leading
        
        # Service and Client Information 
        self.ready_to_compute_gamma = False
        self._impact_backup : dict[int, float] = {}
        self.impact_results : dict[int, float] = {}
        for id in range(1, self.total_agents + 1):
            self.impact_results[id] = 0.0
            self._worst_impact_on_leaders[id] = 0.0
            self._best_impact_from_leaders[id] = 0.0

        # Barriers and tasks
        self.barriers : list[BarrierFunction] = []
        self._barrier_you_are_leading : BarrierFunction = None
        self._barriers_you_are_following : list[BarrierFunction] = []
        self._independent_barrier : BarrierFunction = None
        self.task_msg_list = []
        self.total_tasks = float('inf')

        # Setup publishers
        self.best_impact_pub = self.create_publisher(ImpactMsg, "/best_impact", 100)
        self.worst_impact_pub = self.create_publisher(ImpactMsg, "/worst_impact", 100)
        self.vel_pub = self.create_publisher(Twist, f"/agent{self.agent_id}/cmd_vel", 100)
        self.agent_pose_pub = self.create_publisher(PoseStamped, f"/agent{self.agent_id}/agent_pose", 10)
        self.ready_pub = self.create_publisher(Int32, "/controller_ready", 10)

        # Setup subscribers
        self.create_subscription(ImpactMsg, "/best_impact", self.best_impact_callback, 100)
        self.create_subscription(ImpactMsg, "/worst_impact", self.worst_impact_callback, 100)
        self.create_subscription(LeafNodes, "/leaf_nodes", self.leaf_nodes_callback, 10)
        self.create_subscription(Int32, "/numOfTasks", self.numOfTasks_callback, 10)
        self.create_subscription(TaskMsg, "/tasks", self.task_callback, 10)
        self.create_subscription(LeaderShipTokens, "/tokens", self.tokens_callback, 10)
        for id in range(1, self.total_agents + 1):
            self.create_subscription(PoseStamped, f"/agent{id}/agent_pose", 
                                    partial(self.other_agent_pose_callback, agent_id=id), 10)

        # Setup service server
        self.impact_server = self.create_service(Impact, f"/agent{self.agent_id}/impact_service", self.impact_callback)

        # Setup service clients for each other agent
        self.impact_clients = {}
        for id in range(1, self.total_agents + 1):
            if id != self.agent_id:  # Skip creating a client for itself
                client = self.create_client(Impact, f"/agent{id}/impact_service")
                self.impact_clients[id] = client
                while not client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(f'Impact service for agent{id} not available, waiting...')


        # Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.check_transform_timer = self.create_timer(0.33, self.transform_timer_callback) # 30 Hz = 0.333s
        
        # Wait until all the task messages have been received
        self.task_check_timer = self.create_timer(0.33, self.check_tasks_callback) 


    def impact_callback(self, request, response):

        start_time = time.time()
        self.get_logger().info(f"Received request from agent {request.i} for {request.type} impact")

        if request.type == "best":
            if self._best_impact_on_follower != 0.0:
                response.impact = self._best_impact_on_follower
                self._impact_backup[request.i] = response.impact
                self._best_impact_on_follower = 0.0
                self.get_logger().info(f"sending response to agent {request.i} with best impact value {response.impact}")
            else:
                self.get_logger().info(f"best impact not computed yet, using backup value: {self._impact_backup.get(request.i, 0.0)}")
                response.impact = self._impact_backup.get(request.i, 0.0)

        elif request.type == "worst":
            if self._worst_impact_on_leaders[request.i] != 0.0: 
                response.impact = self._worst_impact_on_leaders[request.i]
                self._impact_backup[request.i] = response.impact
                self._worst_impact_on_leaders[request.i] = 0.0 
                self.get_logger().info(f"sending response to agent {request.i} with worst impact value {response.impact}")
            else:
                self.get_logger().info(f"worst impact not computed yet, using backup value: {self._impact_backup.get(request.i, 0.0)}")
                response.impact = self._impact_backup.get(request.i, 0.0)

        end_time = time.time()
        self.get_logger().info(f"impact_callback execution time for agent {self.agent_id}: {end_time - start_time} seconds")
        return response

        

    def call_impact_service(self, neighbour_id: int, impact_type: str):

        request = Impact.Request()
        request.i = self.agent_id
        request.j = neighbour_id
        request.type = impact_type

        future = self.impact_clients[neighbour_id].call_async(request)
        future.add_done_callback(partial(self.future_callback))
        
        executor = MultiThreadedExecutor()
        executor.add_node(self)

        rclpy.spin_until_future_complete(self, future, executor=executor, timeout_sec=0.2)

        self.get_logger().info(f"This is how the future.result() looks like: {future.result()}")
        
        if future.done() and future.result() is not None:
            self.get_logger().info(f"Received response from agent {neighbour_id} with impact value {future.result().impact}")
            return future.result().impact
        else:
            self.get_logger().error('Service call failed or timed out')
            return None


    def future_callback(self, future):
        self.get_logger().info("trying to get the result from the future_callback...")
        try:
            response = future.result()
            self.get_logger().info(f"Response in future_callback: {response.impact}")
        except Exception as e:
            self.get_logger().info(f"Service call failed: {e}")
        


    def service_loop(self):

        if self.agent_id in self.leaf_nodes:
            self.process_leaf_node()
        else:
            self.process_non_leaf_node()
        

    def process_leaf_node(self):

        self.get_logger().info("computing gamma and best impact")
        self.compute_gamma_and_best_impact_on_leading_task()
        self.get_logger().info(f"sending a request to the follower {self.follower_neighbour} for the worst impact")
        response = self.call_impact_service(self.follower_neighbour, "worst")
        if response is not None:
            self._worst_impact_from_follower = response


    def process_non_leaf_node(self):

        for leader in self.leader_neighbours:
            response = self.call_impact_service(leader, "best")
            if response is not None:
                self._best_impact_from_leaders[leader] = response

        
        self.get_logger().info("all best impacts are received")
        self.compute_gamma_tilde_values()

        if self.ready_to_compute_gamma:
            self.get_logger().info("computing gamma and best impact")
            self.compute_gamma_and_best_impact_on_leading_task()
            self.ready_to_compute_gamma = False
            self.get_logger().info("computing worst impact")
            self.compute_worst_impact_on_following_task()

        if self.follower_neighbour is not None:
            self.get_logger().info(f"sending a request to the follower {self.follower_neighbour} for the worst impact")
            response = self.call_impact_service(self.follower_neighbour, "worst")
            if response is not None:
                self._worst_impact_from_follower = response

        



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
                    neighbour_id = involved_agents[1]
                else:
                    neighbour_id = involved_agents[0]

                normalized_edge = tuple(sorted([self.agent_id, neighbour_id]))

                if self.agent_id == self.LeaderShipTokens_dict[normalized_edge]: 
                    barrier_you_are_leading = barrier

                else:
                    follower_barriers.append(barrier)

        
        return barrier_you_are_leading, follower_barriers, independent_barrier
    

    def get_leader_and_follower_neighbours(self):
        for edge, token in self.LeaderShipTokens_dict.items():

            if self.agent_id in edge:

                if edge[0] == self.agent_id:
                    neighbour_id = int(edge[1])
                else :
                    neighbour_id = int(edge[0])
            
                if token == self.agent_id: # check if you are the leader of the tasks on this edge
                    self.follower_neighbour = neighbour_id
                else:
                    self.leader_neighbours.append(neighbour_id)


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
        """
        barriers_list = []
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
            initial_conditions = [self.agents[i] for i in message.involved_agents]
            barriers_list += [create_barrier_from_task(task=task, initial_conditions=initial_conditions, alpha_function=self.alpha_fun)]
        
        # Create the conjunction of the barriers on the same edge
        barriers_list = self.conjunction_on_same_edge(barriers_list)
        
        return barriers_list


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
        
        if self.follower_neighbour is not None: # if the agent does not have a follower then it does not need to compute the best impact for the follower and won't get the worst impact from the follower
            parameter_list += [ca_tools.entry("epsilon", shape=1)]
        
        parameters = ca_tools.struct_symMX(parameter_list)
        return parameters


    def get_qpsolver(self) -> ca.qpsol:
        """
        This function creates all that is necessary for the optimization problem and creates the optimization solver.

        Returns:
            solver (ca.qpsol): The optimization solver.

        Note:
            The cost function is a quadratic function of the input vector and the slack variables where the slack variables are used to enforce the barrier constraints.
        
        """
        # Create the impact solver that will be used to compute the best and worst impacts on the barriers (NEEDS FIXING)
        self._impact_solver : ImpactSolverLP = ImpactSolverLP(agent=self.agents[self.agent_id], max_velocity=self.max_velocity)

        # get the neighbours of the current agent
        self.get_leader_and_follower_neighbours()
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        self._barrier_you_are_leading, self._barriers_you_are_following, self._independent_barrier  = self._get_splitted_barriers(barriers= self.relevant_barriers)


        # Create the parameter structure for the optimization problem --- 'p' ---
        self.parameters = self.controller_parameters()

        # Create the constraints for the optimization problem --- 'g' ---
        input_constraints = self.A @ self.input_vector - self.parameters["gamma"] * self.b 
        barrier_constraints    = self.generate_barrier_constraints(self.relevant_barriers)
        slack_constraints      = - ca.vertcat(*list(self.slack_variables.values()))
        constraints            = ca.vertcat(input_constraints, barrier_constraints, slack_constraints)

        # Create the decision variables for the optimization problem --- 'x' ---
        slack_vector = ca.vertcat(*list(self.slack_variables.values()))
        opt_vector   = ca.vertcat(self.input_vector, slack_vector)

        # Create the object function for the optimization problem --- 'f' ---
        cost = self.input_vector.T @ self.input_vector
        for id,slack in self.slack_variables.items():
            if id == self.agent_id:
                cost += 100* slack**2 
            else:
                cost += 10* slack**2

        # Create the optimization solver
        qp = {'x': opt_vector, 'f': cost, 'g': constraints, 'p': self.parameters}
        solver = ca.qpsol('sol', 'qpoases', qp, {'printLevel': 'none'})

        return solver


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
                    neighbour_id = barrier.contributing_agents[1]
                else:
                    neighbour_id = barrier.contributing_agents[0]
            else :
                neighbour_id = self.agent_id
            

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

            # check edge for leading agent
            leader = self.LeaderShipTokens_dict[tuple(sorted([self.agent_id, neighbour_id]))]         

            if leader == self.agent_id:
                load_sharing = 1
                barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b) + self.parameters['epsilon'])
            elif leader == neighbour_id:
                load_sharing = 1/2  
                slack = ca.MX.sym(f"slack", 1)
                self.slack_variables[neighbour_id] = slack
                barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b) + slack)     

            constraints.append(barrier_constraint)
            self.barrier_func.append(barrier_fun)
            self.nabla_funs.append(nabla_xi_fun)
            self.nabla_inputs.append(named_inputs)

        return ca.vertcat(*constraints)


    def control_loop(self):
        """This is the main control loop of the controller. It calculates the optimal input and publishes the velocity command to the cmd_vel topic."""
        
        # Get the current time
        time_in_sec,_ = self.get_clock().now().seconds_nanoseconds()
        self.current_time = ca.vertcat(time_in_sec - self.initial_time)

        # Fill the structure with the current state and time
        current_parameters = self.parameters(0)

        # Fill the structure with the values
        current_parameters["time"] = self.current_time
        current_parameters["gamma"] = self._gamma
        for id in self.agents.keys():
            current_parameters[f'state_{id}'] = ca.vertcat(self.agents[id].state[0], self.agents[id].state[1])

        if self.follower_neighbour is not None:
            current_parameters["epsilon"] = self._worst_impact_from_follower


        # Clean the stored values
        self._gamma_tilde = {}
        self._gamma = 1
        self.ready_to_compute_gamma = False


        # Calculate the gradient values to check for convergence
        nabla_list = []
        inputs = {}
        for i, nabla_fun in enumerate(self.nabla_funs):
            inputs = {key: current_parameters[key] for key in self.nabla_inputs[i].keys()}
            self.get_logger().info(f"barrier {i+1} : {self.barrier_func[i].call(inputs)['value']}")
            nabla_val = nabla_fun.call(inputs)["value"]
            nabla_list.append(ca.norm_2(nabla_val))


        # Solve the optimization problem 
        if any(ca.norm_2(val) < 1e-10 for val in nabla_list):
            optimal_input = ca.MX.zeros(2 + len(self.slack_variables))
        else:
            sol = self.solver(p=current_parameters, ubg=0)
            optimal_input = sol['x']
    
        # Publish the velocity command
        linear_velocity = optimal_input[:2]
        clipped_linear_velocity = np.clip(linear_velocity, -self.max_velocity, self.max_velocity)
        self.vel_cmd_msg.linear.x = clipped_linear_velocity[0][0]
        self.vel_cmd_msg.linear.y = clipped_linear_velocity[1][0]
    
        self.vel_pub.publish(self.vel_cmd_msg)
        # self.get_logger().info(f"Published velocity command: {self.vel_cmd_msg}")

        


    def compute_gamma_tilde_values(self):

        self.get_logger().info(f"Trying to compute gamma tilde values...")
        for barrier in self._barriers_you_are_following:
            involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
            
            if involved_agent[0] == self.agent_id:
                neighbour_id = involved_agent[1]
            else :
                neighbour_id = involved_agent[0]
            
            if not (neighbour_id in self._gamma_tilde.keys()):
                gamma_tilde = self._compute_gamma_for_barrier(barrier)
                if gamma_tilde != None:
                    self._gamma_tilde[neighbour_id] = gamma_tilde
        self.get_logger().info(f"gamma tilde values for agent {self.agent_id}: {self._gamma_tilde}")
        if len(self._gamma_tilde) == len(self.leader_neighbours):
            self.ready_to_compute_gamma = True
        else:
            self.get_logger().error(f"Only {len(self._gamma_tilde)} out of {len(self.leader_neighbours)} gamma tilde values are computed.")
        
        
    
    def _compute_gamma_for_barrier(self, barrier: BarrierFunction) -> float :
    
        involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
        
        if len(involved_agent) ==1:
            raise RuntimeError("The given barrier function is a self task. Gamma should not be computed for this task. please revise the logic of the contoller")
        else :
             
            if involved_agent[0] == self.agent_id:
                neighbour_id = involved_agent[1]
            else :
                neighbour_id = involved_agent[0]
                
        
        barrier_fun    : ca.Function   = barrier.function
        local_gradient : ca.Function   = barrier.gradient_function_wrt_state_of_agent(self.agent_id)    
        
        associated_alpha_function :ca.Function  = barrier.associated_alpha_function # this is the alpha function associated to the barrier function in the barrier constraint
        partial_time_derivative:ca.Function     = barrier.partial_time_derivative
        
        if associated_alpha_function == None:
            raise RuntimeError("The alpha function associated to the barrier function is null. please remeber to store this function in the barrier function object for barrier computation")

        neigbour_state       = self.agents[neighbour_id].state    # the neighbour state
        current_agent_state  = self.agents[self.agent_id].state  # your current state
        time                 = self.current_time
        named_inputs         = {"state_"+str(self.agent_id):current_agent_state.flatten(),"state_"+str(neighbour_id):neigbour_state.flatten(),"time":time}

        local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient
        
        g_value = np.eye(current_agent_state.size)

        if local_gardient_value.shape[0] == 1:
            local_gardient_value = local_gardient_value.T

        if neighbour_id in self.leader_neighbours: 
            # this can be parallelised for each of the barrier you are a follower of
            worst_input = self._impact_solver.minimize(Lg = local_gardient_value.T @ g_value) # find the input that minimises the dot product with Lg given the bound on the input
            if worst_input.shape[0] == 1:
                worst_input = worst_input.T
        
            # now we need to compute the gamma for this special case  (recive the best impact from the leader via a topic)
            try :
                neighbour_best_impact          = self._best_impact_from_leaders[neighbour_id] # this is a scalar value representing the best impact of the neigbour on the barrier given its intupt limitations
                self.get_logger().info(f"Upack leader best impact from agent {neighbour_id} with value {neighbour_best_impact}") 
            except:
                self.get_logger().info(f"Required leaders best impact from agent {neighbour_id} not available yet. Retry later...") 
                return None
            
            alpha_barrier_value            = associated_alpha_function(barrier_fun.call(named_inputs)["value"])                # compute the alpha function associated to the barrier function
            partial_time_derivative_value  = partial_time_derivative.call(named_inputs)["value"]                               # compute the partial time derivative of the barrier function
            
            zeta = alpha_barrier_value + partial_time_derivative_value 
            
            if np.linalg.norm(local_gardient_value) <= 10**-6 :
                gamma_tilde = 1
            else : #TODO: Introduce a reduction factor for collision avoidance
                gamma_tilde =  -(neighbour_best_impact + zeta) / ( local_gardient_value.T @ g_value @ worst_input) # compute the gamma value
                
                if alpha_barrier_value < 0 :
                    self.get_logger().warning(f"Alpha barrier value is negative. This entails task dissatisfaction. The value is {alpha_barrier_value}. Please verify that the task is feasible")
                return float(gamma_tilde)
   
    
    def compute_gamma_and_best_impact_on_leading_task(self) : 

        # now it is the time to check if you have all the available information
        if len(self.leader_neighbours) == 0 : # leaf node case # or self.agent_id in self.leaf_nodes
            self._gamma = 1
        elif len(self._gamma_tilde) != len(self.leader_neighbours) :
            raise RuntimeError(f"The list of gamma tilde values is not complete. You have {len(self._gamma_tilde)} values, but you should have {len(self.leader_neighbours)} values from each of the leaders. Make sure you compute the gamma_tilde value for each leader at this iteration.")
        else:
            gamma_tildes_list = list(self._gamma_tilde.values())
            self._gamma = min(gamma_tildes_list + [1]) # take the minimum of the gamma tilde values
            
            if self._gamma<=0 :
                self.get_logger().error(f"The computed gamma value is negative. This breakes the process. The gamma value is {self._gamma}")
                raise RuntimeError(f"The computed gamma value for agent {self.agent_id} is negative. This breakes the process. The gamma value is {self._gamma}")
            
        
        # now that computation of the best impact is undertaken
        if self.follower_neighbour is not None : # if you have a task you are leader of then you should compute your best impact for the follower agent
            # self.get_logger().info(f"Sending Best impact notification to the follower agent {self.follower_neighbour}")
            # now compute your best input for the follower agent
            local_gradient :ca.Function = self._barrier_you_are_leading.gradient_function_wrt_state_of_agent(self.agent_id)    

            named_inputs   = {"state_"+str(self.agent_id)         :self.agents[self.agent_id].state,
                              "state_"+str(self.follower_neighbour):self.agents[self.follower_neighbour].state,
                              "time"                                 :self.current_time}

            local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient
            g_value = np.eye(self.agents[self.agent_id].state.size)

            # then you are leader of this task and you need to compute your best impact
            best_input = self._impact_solver.maximize(Lg= local_gardient_value@g_value) # sign changed because you want the maximum in reality. 
            
            if best_input.shape[0] == 1:
                best_input = best_input.T
            if local_gardient_value.shape[0] == 1:
                local_gardient_value = local_gardient_value.T

            
            best_impact_value = np.dot(local_gardient_value.T,(g_value @ best_input*self._gamma)) # compute the best impact of the leader on the barrier given the gamma_i
            best_impact_value = np.squeeze(best_impact_value)
            self._best_impact_on_follower = float(best_impact_value)




            # Publishing the best impact to the follower
            # best_impact_msg = ImpactMsg()
            # best_impact_msg.i = self.agent_id
            # best_impact_msg.j = self.follower_neighbour
            # best_impact_msg.impact = self._best_impact_on_follower 
            # self.best_impact_pub.publish(best_impact_msg)
            # self.get_logger().info(f"===Published best impact: {self._best_impact_on_follower}")


    
    def compute_worst_impact_on_following_task(self) :
        
        # if you have leaders to notify then do it
        if len(self.leader_neighbours) != 0 :
            # self.get_logger().info(f"Sending worst impact notification to leaders.... ")
            
            for barrier in self._barriers_you_are_following:
                # now compute your best input for the follower agent
                involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
                
                if len(involved_agent) > 1:
                    if involved_agent[0] == self.agent_id:
                        leader_neighbour = involved_agent[1]
                    else :
                        leader_neighbour = involved_agent[0]
                else : # single agent task doesn't need any worst_input computation
                    continue
                    
                
                local_gradient :ca.Function = barrier.gradient_function_wrt_state_of_agent(self.agent_id)    
                
                named_inputs   = {"state_"+str(self.agent_id)        :self.agents[self.agent_id].state,
                                  "state_"+str(leader_neighbour)      :self.agents[leader_neighbour].state ,
                                  "time"                              :self.current_time}

                local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient

                g_value = np.eye(self.agents[self.agent_id].state.size)

                # then you are leader of this task and you need to compute your best impact
                worst_input = self._impact_solver.minimize(Lg= local_gardient_value@g_value) 
                
                
                if worst_input.shape[0] == 1:
                    worst_input = worst_input.T
                if local_gardient_value.shape[0] == 1:
                    local_gardient_value = local_gardient_value.T

                worst_impact_value = np.dot(local_gardient_value.T,(g_value @ worst_input*self._gamma))
                worst_impact_value = np.squeeze(worst_impact_value)
                self._worst_impact_on_leaders[leader_neighbour] = float(worst_impact_value)
                self.get_logger().info(f"worst impact for leader agent {leader_neighbour} is {self._worst_impact_on_leaders[leader_neighbour]}")

                # Publishing the worst impact to the leaders
                # worst_impact_msg = ImpactMsg()
                # worst_impact_msg.i = self.agent_id
                # worst_impact_msg.j = leader_neighbour
                # worst_impact_msg.impact = self._worst_impact_on_leaders[leader_neighbour]
                # self.worst_impact_pub.publish(worst_impact_msg)
                # self.get_logger().info(f"===Published worst impact for leader agent {leader_neighbour} with value: {self._worst_impact_on_leaders[leader_neighbour]}")
 
            









    #  ==================== Callbacks ====================

    def other_agent_pose_callback(self, msg, agent_id):
        """
        Callback function to store all the agents' poses.
        
        Args:
            msg (PoseStamped): The pose message of the other agents.
            agent_id (int): The ID of the agent extracted from the topic name.

        """
        state = np.array([msg.pose.position.x, msg.pose.position.y])
        self.agents[agent_id] = Agent(id=agent_id, initial_state=state)


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
            self.get_logger().info("==All tasks received==")
            self.initial_time,_ = self.get_clock().now().seconds_nanoseconds()
            
            # Create the tasks and the barriers
            self.barriers = self.create_barriers_from_tasks(self.task_msg_list)
            self.solver = self.get_qpsolver()
            self.service_timer = self.create_timer(0.33, self.service_loop)
            self.control_loop_timer = self.create_timer(0.33, self.control_loop)
        else:
            ready = Int32()
            ready.data = self.agent_id
            self.ready_pub.publish(ready)


    def transform_timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform("world", "nexus_"+self.agent_name, Time())
            # update self tranform
            self.latest_self_transform = trans

            # Send your position to the other agents
            position_msg = PoseStamped()
            position_msg.header.stamp = self.get_clock().now().to_msg()
            position_msg.pose.position.x = trans.transform.translation.x
            position_msg.pose.position.y = trans.transform.translation.y
            self.agent_pose_pub.publish(position_msg)

        except LookupException as e:
            self.get_logger().error('failed to get transform {} \n'.format(repr(e)))


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


    def best_impact_callback(self, msg):
        """
        Callback function for the best impact message.
        
        Args:
            msg (ImpactMsg): The best impact message.
        """
        if msg.i in self.leader_neighbours:
            self._best_impact_from_leaders[msg.i] = msg.impact
            # self.get_logger().info(f"Received best impact from agent {msg.i} with value {msg.impact}")
        else:
            pass 

    
    def worst_impact_callback(self, msg):
        """
        Callback function that stores the worst impact value which agent i calculated for the task that agent j is leading.
        
        Args:
            msg (ImpactMsg): The worst impact message.
        """
        if msg.j == self.agent_id:
            self._worst_impact_from_follower = msg.impact
            # self.get_logger().info(f"Received worst impact from agent {msg.i} with value {msg.impact}")
        else:
            pass
        

        

    

def main(args=None):
    rclpy.init(args=args)

    node = Controller()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()