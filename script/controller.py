#!/usr/bin/env python3
import sys
import copy
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import numpy as np
import casadi as ca
import tf2_geometry_msgs
from typing import List, Dict
from std_msgs.msg import Int32
import casadi.tools as ca_tools
from collections import defaultdict
from custom_msg.msg import task_msg
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Vector3Stamped
from builders import (BarrierFunction, Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, 
                      create_barrier_from_task, go_to_goal_predicate_2d, formation_predicate, 
                      epsilon_position_closeness_predicate, conjunction_of_barriers, collision_avoidance_predicate)


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
        last_pose_time         (rospy.Time)               : The timestamp of the last received agent pose.
        agents                 (dict)                     : A dictionary containing all agents and their states.
        total_agents           (int)                      : The total number of agents in the environment.
        barriers               (list)                     : A list of barrier functions used to construct the constraints of the optimization problem.
        task                   (task_msg)                 : The message that contains the task information.
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
        self.solver = None
        self.parameters = None
        self.input_vector = ca.MX.sym('input', 2)
        self.slack_variables = {}
        self.scale_factor = 3
        self.dummy_scalar = ca.MX.sym('dummy_scalar', 1)
        self.alpha_fun = ca.Function('alpha_fun', [self.dummy_scalar], [self.scale_factor * self.dummy_scalar])
        self.nabla_funs = []
        self.nabla_inputs = []

        # Agent Information
        self.agent_pose = PoseStamped()
        self.agent_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.agent_id = int(self.agent_name[-1])
        self.last_pose_time = Time()

        # Neighbouring Agents
        self.agents = {}
        self.total_agents = self.get_parameter('num_robots').get_parameter_value().integer_value
        
        # Barriers and tasks
        self.barriers = []
        self.task = task_msg()
        self.task_msg_list = []
        self.total_tasks = float('inf')

        # Velocity Command Message
        self.max_velocity = 5
        self.vel_cmd_msg = Twist()

        # Setup subscribers
        self.create_subscription(Int32, "/numOfTasks", self.numOfTasks_callback, 10)
        self.create_subscription(task_msg, "/tasks", self.task_callback, 10)
        self.create_subscription(PoseStamped, f"/qualisys/{self.agent_name}/pose", self.pose_callback, 10)
        for id in range(1, self.total_agents + 1):
            self.create_subscription(PoseStamped, f"/nexus{id}/agent_pose", self.other_agent_pose_callback, 10)

        # Setup publishers
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 100)
        self.agent_pose_pub = self.create_publisher(PoseStamped, f"agent_pose", 100)

        # Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.tf_buffer.can_transform('mocap', self.agent_name, Time()):
            self.get_logger().info("Wait for transform to be available")
            rclpy.sleep(1)


        # Wait until all the task messages have been received
        while len(self.task_msg_list) < self.total_tasks:
            # self.get_logger().info(f"Waiting for all tasks to be received. Received {len(self.task_msg_list)} out of {self.total_tasks}")
            rclpy.sleep(1)

        # Create the tasks and the barriers
        self.barriers = self.create_barriers(self.task_msg_list)
        self.solver = self.get_qpsolver_and_parameter_structure()
        self.control_loop()


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


    def create_barriers(self, messages:List[task_msg]) -> List[BarrierFunction]:
        """
        Constructs the barriers from the subscribed task messages.     
        
        Args:
            messages (List[task_msg]): List of task messages.
        
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
            elif message.type == "collision_avoidance_predicate":
                predicate = collision_avoidance_predicate(epsilon=message.epsilon, agent_i=self.agents[message.involved_agents[0]], 
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


    def get_qpsolver_and_parameter_structure(self) -> ca.qpsol:
        """
        Initializes the parameter structure that contains the state of the agents and the time. 
        This function creates all that is necessary for the optimization problem and creates the optimization solver.

        Returns:
            solver (ca.qpsol): The optimization solver.

        Note:
            The cost function is a quadratic function of the input vector and the slack variables where the slack variables are used to enforce the barrier constraints.
            The slack variables are multiplied by a factor of 1000 for the leading agent and 10 for the other agents in an attempt to prioritize self-tasks.
        
        """

        # Create the parameter structure for the optimization problem --- 'p' ---
        parameter_list  = []
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in self.agents.keys()]
        parameter_list += [ca_tools.entry("time", shape=1)]
        self.parameters = ca_tools.struct_symMX(parameter_list)

        # Create the constraints for the optimization problem --- 'g' ---
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
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
                cost += 1000* slack**2
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
            named_inputs = {"state_"+str(id):self.parameters["state_"+str(id)] for id in barrier.contributing_agents}
            named_inputs["time"] = self.parameters["time"]

            # Get the necessary functions from the barrier
            nabla_xi_fun                = barrier.gradient_function_wrt_state_of_agent(self.agent_id)
            barrier_fun                 = barrier.function
            partial_time_derivative_fun = barrier.partial_time_derivative

            # Calculate the symbolic expressions for the barrier constraint
            nabla_xi = nabla_xi_fun.call(named_inputs)["value"]
            dbdt     = partial_time_derivative_fun.call(named_inputs)["value"]
            alpha_b  = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"])

            # Create load sharing for different constraints
            if neighbour_id == self.agent_id:
                slack = ca.MX.sym(f"slack", 1)
                self.slack_variables[self.agent_id] = slack
                load_sharing = 1
            else:
                slack = ca.MX.sym(f"slack", 1)
                self.slack_variables[neighbour_id] = slack  
                load_sharing = 0.1

            barrier_constraint = -1 * (ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b + slack))
            constraints.append(barrier_constraint)
            self.nabla_funs.append(nabla_xi_fun)
            self.nabla_inputs.append(named_inputs)

        return ca.vertcat(*constraints)


    def control_loop(self):
        """This is the main control loop of the controller. It calculates the optimal input and publishes the velocity command to the cmd_vel topic."""
        
        rate = self.create_rate(50)
        while rclpy.ok():
            # Fill the structure with the current values
            current_parameters = self.parameters(0)
            current_parameters["time"] = ca.vertcat(self.last_pose_time.nanoseconds / 1e9)
            for id in self.agents.keys():
                current_parameters[f'state_{id}'] = ca.vertcat(self.agents[id].state[0], self.agents[id].state[1])

            # Calculate the gradient values
            nabla_list = []
            inputs = {}
            for i, nabla_fun in enumerate(self.nabla_funs):
                inputs = {key: current_parameters[key] for key in self.nabla_inputs[i].keys()}
                nabla_val = nabla_fun.call(inputs)["value"]
                nabla_list.append(ca.norm_2(nabla_val))

            # Solve the optimization problem 
            if any(ca.norm_2(val) < 1e-10 for val in nabla_list):
                optimal_input = ca.MX([0, 0])
            else:
                sol = self.solver(p=current_parameters, ubg=0)
                optimal_input = sol['x']

            # Publish the velocity command
            self.vel_cmd_msg.linear.x = np.clip(optimal_input[0], -self.max_velocity, self.max_velocity)
            self.vel_cmd_msg.linear.y = np.clip(optimal_input[1], -self.max_velocity, self.max_velocity)

            try:
                # Get transform from mocap frame to agent frame
                transform = self.tf_buffer.lookup_transform('mocap', self.agent_name, Time())
                vel_cmd_msg_transformed = self.transform_twist(self.vel_cmd_msg, transform)
                self.vel_pub.publish(vel_cmd_msg_transformed)
            except tf2_ros.TransformException as e:
                self.get_logger().error(f"Transform error: {e}")
                continue

            rate.sleep()


    # ----------- Callbacks -----------------
    def pose_callback(self, msg):
        """
        Callback function for the agent's pose.

        Args:
            msg (PoseStamped): The pose message of the agent.

        Note:
            This function is used to get the agent pose from the qualisys system and publish it to the agent_pose topic for easy access.
         
        """
        self.agent_pose = msg
        self.agent_pose_pub.publish(self.agent_pose)

    def other_agent_pose_callback(self, msg):
        """
        Callback function to store all the agents' poses.
        
        Args:
            msg (PoseStamped): The pose message of the other agents.

        Note:
            The way I am extracting the agent id is dependent on the topic name and needs to be adjusted if the topic name changes.
        
        """
        agent_id              = int(msg._connection_header['topic'].split('/')[-2].replace('nexus', ''))
        state                 = np.array([msg.pose.position.x, msg.pose.position.y])
        self.agents[agent_id] = Agent(id=agent_id, initial_state=state)
        self.last_pose_time   = Time.now() 

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
            msg (task_msg): The task message.
        """
        self.task_msg_list.append(msg)


def transform_twist(twist=Twist, transform_stamped=TransformStamped) -> Twist:
    """
    Transforms the twist from one frame to another frame.
    
    Args:
        twist             (Twist)            : The cmd_vel message to be transformed.
        transform_stamped (TransformStamped) : The transform between the frames.

    Returns:
        new_twist (Twist) : The transformed cmd_vel message.
    """
    transform_stamped_ = copy.deepcopy(transform_stamped)
    # Inverse real-part of quaternion to inverse rotation
    transform_stamped_.transform.rotation.w = - transform_stamped_.transform.rotation.w

    twist_vel = Vector3Stamped()
    twist_rot = Vector3Stamped()
    twist_vel.vector = twist.linear
    twist_rot.vector = twist.angular
    out_vel = tf2_geometry_msgs.do_transform_vector3(twist_vel, transform_stamped_)
    out_rot = tf2_geometry_msgs.do_transform_vector3(twist_rot, transform_stamped_)

    new_twist = Twist()
    new_twist.linear = out_vel.vector
    new_twist.angular = out_rot.vector

    return new_twist


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