#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
from functools import partial
import numpy as np
import casadi as ca
from typing import List, Dict, Tuple
from std_msgs.msg import Int32, Bool
import casadi.tools as ca_tools
from collections import defaultdict
from stl_decomposition_msgs.msg import TaskMsg, LeaderShipTokens, LeafNodes
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from .dynamics_module import Agent, LeadershipToken, ImpactSolverLP
from .builders import (BarrierFunction, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, 
                      create_barrier_from_task, go_to_goal_predicate_2d, formation_predicate, 
                      epsilon_position_closeness_predicate, conjunction_of_barriers)
from tf2_ros import LookupException


class Controller(Node):
    """
    This class is a STL-QP (Signal Temporal Logic-Quadratic Programming) controller for omnidirectional robots and therefore does not consider the orientation of the robots.
    It is responsible for solving the optimization problem and publishing the velocity commands to the agents. 
    

    Attributes:
        solver                 (ca.Opti)                  : An optimization problem solver instance used for solving the STL-QP optimization problem.
        parameters             (ca_tools.struct_symMX)    : A structure containing parameters necessary for the optimization problem.
        input_vector           (ca.MX)                    : A symbolic variable representing the input vector for the optimization problem.
        slack_variables        (dict)                     : A dictionary containing slack variables used to enforce barrier constraints in the optimization problem.
        scale_factor           (int)                      : A scaling factor applied to the optimization problem to adjust the cost function.
        dummy_scalar           (ca.MX)                    : A dummy scalar symbolic variable used in the optimization problem.
        alpha_fun              (ca.Function)              : A CasADi function representing the scaling factor applied to the dummy scalar in the optimization problem.
        nabla_funs             (list)                     : A list of functions representing the gradients of barrier functions in the optimization problem.
        nabla_inputs           (list)                     : A list of inputs required for evaluating the gradient functions.
        agent_pose             (PoseStamped)              : The current pose of the agent.
        agent_name             (str)                      : The name of the agent.
        agent_id               (int)                      : The ID of the agent.
        agents                 (dict)                     : A dictionary containing all agents and their states.
        total_agents           (int)                      : The total number of agents in the environment.
        barriers               (list)                     : A list of barrier functions used to construct the constraints of the optimization problem.
        task                   (TaskMsg)                  : The message that contains the task information.
        task_msg_list          (list)                     : A list of task messages.
        total_tasks            (float)                    : The total number of tasks that was sent by the manager.
        max_velocity           (int)                      : The maximum velocity of the agent used to limit the velocity command.
        vel_cmd_msg            (Twist)                    : The velocity command message to be published to the cmd_vel topic.
        vel_pub                (rospy.Publisher)          : A publisher for sending velocity commands.
        agent_pose_pub         (rospy.Publisher)          : A publisher for publishing the agent's pose.
        tf_buffer              (tf2_ros.Buffer)           : A buffer for storing transforms.
        tf_listener            (tf2_ros.TransformListener): A listener for receiving transforms.

    Note:
        The controller subscribes to several topics to receive updates about tasks, agent poses, and the number of tasks.
        It also publishes the velocity command and the agent's pose.
        The controller waits until it receives all task messages before creating the tasks and barriers and starting the control loop.
    """

    def __init__(self):
        # Initialize the node
        super().__init__('controller')

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
        self._gamma : float = 1
        self._gamma_tilde :dict[int,float] = {}

        # parameters declaration
        self.declare_parameter('robot_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('num_robots', rclpy.Parameter.Type.INTEGER)

        # Agent Information
        self.agent_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.agent_id = int(self.agent_name[-1])
        self.latest_self_transform = TransformStamped()
        self.total_agents = self.get_parameter('num_robots').get_parameter_value().integer_value
        self.agents : dict[int, Agent]= {} # position of all the agents in the system including self agent
        
        # information stored about your neighbours
        self.LeaderShipTokens_dict : Dict[tuple,int]= {}
        self._leadership_tokens : Dict[int,LeadershipToken] = {}
        self.follower_neighbour : int = None
        self.leader_neighbours : list[int] = []
        self.leaf_nodes : list[int] = []
        
        self._task_neighbours_id         : list[int]                    = []    # list of neigbouring agents IDS in the communication graph

        self._best_impact_from_leaders   : dict[int,float] = {} # for each leader you will receive a best impact that you can use to compute your gamma
        self._worse_impact_on_leaders    : dict[int,float] = {} # after you compute gamma you send back your worse impact to the leaders
        
        self._worse_impact_from_follower : float = 0. # this is the worse impact that the follower of your task will send you back
        self._best_impact_on_follower    : float = 0. # this is the best impact that you can have on thhe task you are leading



        # Barriers and tasks
        self.barriers : list[BarrierFunction] = []
        # self.task = TaskMsg()   # Custom message type from stl_decomposition_msgs
        self.task_msg_list = []
        self.total_tasks = float('inf')

        # Velocity Command Message
        self.max_velocity = 1.0
        self.vel_cmd_msg = Twist()

        # Setup publishers
        self.vel_pub = self.create_publisher(Twist, f"/agent{self.agent_id}/cmd_vel", 100)
        self.agent_pose_pub = self.create_publisher(PoseStamped, f"/agent{self.agent_id}/agent_pose", 10)
        self.ready_pub = self.create_publisher(Bool, "/controller_ready", 10)

        # Setup subscribers
        self.create_subscription(LeafNodes, "/leaf_nodes", self.leaf_nodes_callback, 10)
        self.create_subscription(Int32, "/numOfTasks", self.numOfTasks_callback, 10)
        self.create_subscription(TaskMsg, "/tasks", self.task_callback, 10)
        self.create_subscription(LeaderShipTokens, "/tokens", self.tokens_callback, 10)
        for id in range(1, self.total_agents + 1):
            self.create_subscription(PoseStamped, f"/agent{id}/agent_pose", 
                                    partial(self.other_agent_pose_callback, agent_id=id), 10)

        # Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.check_transform_timer = self.create_timer(0.33, self.transform_timer_callback) # 30 Hz = 0.333s
        # Wait until all the task messages have been received
        self.task_check_timer = self.create_timer(0.5, self.check_tasks_callback) 

    
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

    
    # NEED TO FIX THIS FUNCTION
    def _get_splitted_barriers(self,barriers:List[BarrierFunction]) -> Tuple[BarrierFunction,List[BarrierFunction],BarrierFunction]: #,tokens : Dict[tuple,int]
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

                if self.agent_id == self.LeaderShipTokens_dict[normalized_edge]: # tokens[tuple(sorted([self.agent_id, neighbour_id]))]
                    barrier_you_are_leading = barrier
                else:
                    follower_barriers.append(barrier)
        
        return barrier_you_are_leading, follower_barriers, independent_barrier
    

    def get_leader_and_follower_neighbours(self):
        for edge, token in self.LeaderShipTokens_dict.items():

            if self.agent_id in edge:

                if edge[0] == self.agent_id:
                    neighbour_id = edge[1]
                else :
                    neighbour_id = edge[0]
            
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
            barriers_list += [create_barrier_from_task(task=task, initial_conditions=initial_conditions, alpha_function=self.alpha_fun, t_init=self.initial_time)]
        
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
        if self.follower_neighbour != None: # if the agent does not have a 
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
            The slack variables are multiplied by a factor of 1000 for the leading agent and 10 for the other agents in an attempt to prioritize self-tasks.
        
        """
        # Create the impact solver that will be used to compute the best and worst impacts on the barriers
        self._impact_solver : ImpactSolverLP = ImpactSolverLP(agent=self.agents[self.agent_id])

        self.get_leader_and_follower_neighbours()

        # Create the parameter structure for the optimization problem --- 'p' ---
        self.parameters = self.controller_parameters()

        # Create the constraints for the optimization problem --- 'g' ---
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        self._barrier_you_are_leading, self._barriers_you_are_following, self._independent_barrier  = self._get_splitted_barriers(barriers= self.barriers) # ,tokens = self.LeaderShipTokens_dict
        # self.compute_gamma_tilde_values()
        barrier_constraints    = self.generate_barrier_constraints(self.relevant_barriers)
        slack_constraints      = - ca.vertcat(*list(self.slack_variables.values()))
        constraints            = ca.vertcat(barrier_constraints, slack_constraints)

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
        print("=====================================")
        print(f"leadership token: {self.LeaderShipTokens_dict.keys()},{self.LeaderShipTokens_dict.values()}")
        print("=====================================")
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
            print("=====================================")
            print(f"leader: {leader}")
            

            if leader == self.agent_id:
                load_sharing = 1
                barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b)) # + worst impact from follower
            elif leader == neighbour_id:
                load_sharing = 1/2  # 1/len(barrier.contributing_agents)
                slack = ca.MX.sym(f"slack", 1)
                self.slack_variables[neighbour_id] = slack
                barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b + slack))     

            constraints.append(barrier_constraint)
            self.barrier_func.append(barrier_fun)
            self.nabla_funs.append(nabla_xi_fun)
            self.nabla_inputs.append(named_inputs)

        return ca.vertcat(*constraints)


    def control_loop(self):
        """This is the main control loop of the controller. It calculates the optimal input and publishes the velocity command to the cmd_vel topic."""
        # Fill the structure with the current state and time
        current_parameters = self.parameters(0)
        time_in_sec,_ = self.get_clock().now().seconds_nanoseconds()
        current_parameters["time"] = ca.vertcat(time_in_sec-self.initial_time)
        for id in self.agents.keys():
            current_parameters[f'state_{id}'] = ca.vertcat(self.agents[id].state[0], self.agents[id].state[1])

        # Calculate the gradient values to check for convergence
        nabla_list = []
        inputs = {}
        for i, nabla_fun in enumerate(self.nabla_funs):
            inputs = {key: current_parameters[key] for key in self.nabla_inputs[i].keys()}
            print(f"barrier {i+1} : {self.barrier_func[i].call(inputs)['value']}")
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
        







    # ==================== Modify and clean up the following functions ====================

    def compute_gamma_tilde_values(self):
        time_in_sec,_ = self.get_clock().now().seconds_nanoseconds()
        current_time = ca.vertcat(time_in_sec-self.initial_time)
        
        self._logger.info(f"Trying to compute gamma tilde values...")
        for barrier in self._barriers_you_are_following:
            involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
            
            if involved_agent[0] == self.agent_id:
                neighbour_id = involved_agent[1]
            else :
                neighbour_id = involved_agent[0]
            
            if not (neighbour_id in self._gamma_tilde.keys()):
                gamma_tilde = self._compute_gamma_for_barrier(barrier, current_time)
                if gamma_tilde != None:
                    self._gamma_tilde[neighbour_id] = gamma_tilde
        
        # if self._has_self_tasks :
        #     # here we fill the nu for the self task. the other self._nu are computed inside the _compute_gamma_for_barrrier
        #     self._nu[self.agent_id] = self._personal_upsilon_values[self.agent_id] 
        
    
    def _compute_gamma_for_barrier(self, barrier: BarrierFunction, current_time: float) -> float :
    
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
        time                 = current_time
        named_inputs         = {"state_"+str(self.agent_id):current_agent_state.flatten(),"state_"+str(neighbour_id):neigbour_state.flatten(),"time":time}

        local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient
        
        g_value = np.eye(current_agent_state.size)

        if local_gardient_value.shape[0] == 1:
            local_gardient_value = local_gardient_value.T

        if neighbour_id in self.leader_neighbours: 
            # this can be parallelised for each of the barrier you are a follower of
            worse_input = self._impact_solver.minimize(Lg = local_gardient_value.T @ g_value) # find the input that minimises the dot product with Lg given the bound on the input
            if worse_input.shape[0] == 1:
                worse_input = worse_input.T

            # now we need to compute the gamma for this special case  (recive the best impact from the leader via a topic)
            try :
                neighbour_best_impact          = self._best_impact_from_leaders[neighbour_id] # this is a scalar value representing the best impact of the neigbour on the barrier given its intupt limitations
                self._logger.info(f"Upack leader best impact from agent {neighbour_id} with value {neighbour_best_impact}") 
            except:
                self._logger.info(f"Required leaders best impact from agent {neighbour_id} not available yet. Retry later...") 
                return None
            
            alpha_barrier_value            = associated_alpha_function(barrier_fun.call(named_inputs)["value"])                # compute the alpha function associated to the barrier function
            partial_time_derivative_value  = partial_time_derivative.call(named_inputs)["value"]                               # compute the partial time derivative of the barrier function
            
            # nu   = self._upsilon_from_neigbours[neighbour_id] + self._personal_upsilon_values[neighbour_id]
            zeta = alpha_barrier_value + partial_time_derivative_value 
            # self._nu[neighbour_id] = nu # store the value of nu

            
            if np.linalg.norm(local_gardient_value) <= 10**-6 :
                gamma_tilde = 1
            else :
                # if self._enable_collision_avoidance :
                #     reduction_factor = 0.8 # when you have collsion avoidance you should assume that the leader might not be able to actually express its best input because he is busy avoiding someone else. So use a reduction factor
                # else :
                #     reduction_factor = 1
                reduction_factor = 1
                gamma_tilde =  -(neighbour_best_impact*reduction_factor + zeta) / ( local_gardient_value.T @ g_value @ worse_input) # compute the gamma value
                
                if alpha_barrier_value < 0 :
                    self._logger.warning(f"Alpha barrier value is negative. This entails task dissatisfaction. The value is {alpha_barrier_value}. Please very fy that the task is feasible")
                return float(gamma_tilde)
   
    
    def compute_gamma_and_notify_best_impact_on_leader_task(self, current_time: float) :
        
        # now it is the time to check if you have all the available information
        if len(self.leader_neighbours) == 0 : # leaf node case
            self._gamma = 1
        elif len(self._gamma_tilde) != len(self.leader_neighbours) :
            raise RuntimeError(f"The list of gamma tilde values is not complete. You have {len(self._gamma_tilde)} values, but you should have {len(self.leader_neighbours)} values from each of the leaders. Make sure you compute the gamma_tilde value for each leader at this iteration.")
        
        else :
            
            gamma_tildes_list = list(self._gamma_tilde.values())
            self._gamma = min(gamma_tildes_list + [1]) # take the minimum of the gamma tilde values
            
            if self._gamma<=0 :
                self._logger.error(f"The computed gamma value is negative. This breakes the process. The gamma value is {self._gamma}")
                raise RuntimeError(f"The computed gamma value for agent {self.agent_id} is negative. This breakes the process. The gamma value is {self._gamma}")
            
        
        # now that computation of the best impact is undertaken
        if self.follower_neighbour != None : # if you have a task you are leader of then you should compute your best impact for the follower agent
            self._logger.info(f"Sending Best impact notification to the follower agent {self.follower_neighbour}")
            # now compute your best input for the follower agent
            local_gradient :ca.Function = self._barrier_you_are_leading.gradient_function_wrt_state_of_agent(self.agent_id)    

            named_inputs   = {"state_"+str(self.agent_id)         :self.agents[self.agent_id].state,
                              "state_"+str(self.follower_neighbour):self.agents[self.follower_neighbour].state,
                              "time"                                 :current_time}

            local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient
            g_value = np.eye(self.agents[self.agent_id].state.size)

            
            # then you are leader of this task and you need to compute your best impact
            best_input = self._impact_solver.maximize(Lg= local_gardient_value@g_value) # sign changed because you want the maximum in reality. 
            
            if best_input.shape[0] == 1:
                best_input = best_input.T
            if local_gardient_value.shape[0] == 1:
                local_gardient_value = local_gardient_value.T

            self._best_impact_on_leader_task   = np.dot(local_gardient_value.T,(g_value @ best_input*self._gamma)) # compute the best impact of the leader on the barrier given the gamma_i
            self._best_impact_on_leader_task   = np.squeeze(self._best_impact_on_leader_task)
            
            # store the best impact (maybe send it to the follower via a topic instead)
            self._best_impact_from_leaders[self.agent_id] = self._best_impact_on_leader_task 


            # save nu for this function
            # associated_alpha_function :ca.Function  = self._barrier_you_are_leading.associated_alpha_function # this is the alpha function associated to the barrier function in the barrier constraint
            # partial_time_derivative:ca.Function     = self._barrier_you_are_leading.partial_time_derivative
            
            # alpha_barrier_value            = associated_alpha_function(self._barrier_you_are_leading.function.call(named_inputs)["value"])                # compute the alpha function associated to the barrier function
            # partial_time_derivative_value  = partial_time_derivative.call(named_inputs)["value"]   
            
            # zeta = alpha_barrier_value + partial_time_derivative_value


    
    def compute_and_notify_worse_impact_on_follower_task(self,current_time: float) :
        
        # if you have leaders to notify then do it
        if len(self.leader_neighbours) != 0 :
            self._logger.info(f"Sending worse impact notification to leaders.... ")
            
            for barrier in self._barriers_you_are_following:
                # now compute your bst input for the follower agent
                involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
                
                if len(involved_agent) > 1:
                    if involved_agent[0] == self.agent_id:
                        leader_neighbour = involved_agent[1]
                    else :
                        leader_neighbour = involved_agent[0]
                else : # single agent task doesn't need any worse_input computation
                    continue
                    
                
                local_gradient :ca.Function = barrier.gradient_function_wrt_state_of_agent(self.agent_id)    
                
                named_inputs   = {"state_"+str(self.agent_id)        :self.agents[self.agent_id].state,
                                  "state_"+str(leader_neighbour)      :self.agents[leader_neighbour].state ,
                                  "time"                              :current_time}

                local_gardient_value = local_gradient.call(named_inputs)["value"] # compute the local gradient

                g_value = np.eye(self.agents[self.agent_id].state.size)

                # then you are leader of this task and you need to compute your best impact
                worse_input = self._impact_solver.minimize(Lg= local_gardient_value@g_value) 
                
                
                if worse_input.shape[0] == 1:
                    worse_input = worse_input.T
                if local_gardient_value.shape[0] == 1:
                    local_gardient_value = local_gardient_value.T

                # send your worse impact to the leaders (via a topic)
                self._worse_impact_on_leaders[leader_neighbour]  = np.dot(local_gardient_value.T,(g_value @ worse_input*self._gamma)) # compute the best impact of the leader on the barrier given the gamma_i
                
    # ========================================================================












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
        if len(self.task_msg_list) >= self.total_tasks:
            self.task_check_timer.cancel()  # Stop the timer if all tasks are received
            self.get_logger().info("All tasks received.")
            self.initial_time,_ = self.get_clock().now().seconds_nanoseconds()
            
            # Create the tasks and the barriers
            self.barriers = self.create_barriers_from_tasks(self.task_msg_list)
            self.solver = self.get_qpsolver()
            self.control_loop_timer = self.create_timer(0.5, self.control_loop)
        else:
            ready = Bool()
            ready.data = True
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






    def leaf_nodes_callback(self, msg):
        """
        Callback function to get the leaf nodes in the communication graph.

        Args:
            msg (LeafNodes): The leaf nodes message.
        """
        self.leaf_nodes = msg.list
        

        

    

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