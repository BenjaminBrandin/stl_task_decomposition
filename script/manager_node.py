#!/usr/bin/env python3
import os
import yaml
import rclpy
import multiprocessing as mp
from rclpy.node import Node
import numpy as np
import networkx as nx
from   networkx import diameter as net_diameter
import casadi as ca
from typing import Dict
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
from stl_decomposition_msgs.msg import TaskMsg, EdgeLeaderShip, LeaderShipTokens, LeafNodes
from ament_index_python.packages import get_package_share_directory
from .decomposition_module import computeNewTaskGraph
from .graph_module import create_communication_graph_from_states, create_task_graph_from_edges, fixed_communication_graph
from .dynamics_module import Agent, LeadershipToken
from .builders import (StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, 
                      formation_predicate, epsilon_position_closeness_predicate)

Ti = Dict[int,LeadershipToken] # token set fo a single agent

class Manager(Node):

    """
    Manages the creation of task and communication graphs and performs task decomposition on non-communicative edges.
    
    This class reads initial states and tasks from YAML files.

    Attributes:
        agents             (dict[int, Agent])      : Dictionary of agents.
        total_tasks        (int)                   : Total number of tasks.
        start_pos          (dict[int, np.ndarray]) : Initial positions loaded from YAML file.
        tasks              (dict[str, dict])       : Tasks loaded from YAML file.
        task_pub           (rospy.Publisher)       : Publisher for tasks.
        numOfTasks_pub     (rospy.Publisher)       : Publisher for the number of tasks.
        comm_graph         (networkx.Graph)        : Communication graph.
        task_graph         (networkx.Graph)        : Task graph.
        initial_task_graph (networkx.Graph)        : Initial task graph.

    """

    def __init__(self):
        """Initializes a Manager object."""

        super().__init__('manager')

        self.agents: dict[int, Agent] = {}
        self.total_tasks: int = 0
        communication_radius: float = 4.0
        self.fixed_communication_flag: bool = False
        self.bool_msg :bool = False
        self.ready_controllers : list[int] = []

        # setup publishers
        self.task_pub = self.create_publisher(TaskMsg, "/tasks", 100)
        self.numOfTasks_pub = self.create_publisher(Int32, "/numOfTasks", 10)
        self.tokens_pub = self.create_publisher(LeaderShipTokens, "/tokens", 100)
        self.leaf_nodes_pub = self.create_publisher(LeafNodes, "/leaf_nodes", 10)

        # setup subscribers
        self.create_subscription(Int32, "/controller_ready", self.controller_ready_callback, 10)

        # Define package names and subpaths
        pkg_name = 'stl_task_decomposition'
        config_folder_subpath = 'config'
        
        # Get the paths to the YAML files
        start_pos_yaml_path = os.path.join(
            get_package_share_directory(pkg_name), config_folder_subpath, 'initial_conditions.yaml')
        tasks_yaml_path = os.path.join(
            get_package_share_directory(pkg_name), config_folder_subpath, 'tasks.yaml')
        fixed_communication_yaml_path = os.path.join(
            get_package_share_directory(pkg_name), config_folder_subpath, 'fixed_communications.yaml')

        # Load the initial states and the tasks from the YAML files
        with open(start_pos_yaml_path, 'r') as file:
            self.start_pos = yaml.safe_load(file)

        with open(tasks_yaml_path, 'r') as file:
            self.tasks = yaml.safe_load(file)

        with open(fixed_communication_yaml_path, 'r') as file:
            self.fixed_communications = yaml.safe_load(file)

        # Create the Agent objects and store them in a dictionary
        start_positions = {}
        initial_conditions = self.start_pos['initial_conditions']
        for agent_name, state in initial_conditions.items():
            agent_id = int(agent_name.replace('agent', ''))
            position = np.array([state['x'], state['y']])
            self.agents[agent_id] = Agent(id=agent_id, initial_state=position)
            start_positions[agent_id] = position
            
        # Extracting the edges of the tasks
        task_edges = [tuple(task["EDGE"]) for task in self.tasks.values()]

        # Creating the graphs
        communication_edges = [tuple(edge["agents"]) for edge in self.fixed_communications.values()]

        if self.fixed_communication_flag:
            self.comm_graph = fixed_communication_graph(start_positions, communication_edges)  # creates a communication graph using yaml file with fixed edges.
        else:
            self.comm_graph = create_communication_graph_from_states(start_positions, communication_radius)  # creates a communication graph using yaml file with fixed edges.
        
        

        self.task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
        self.initial_task_graph = self.task_graph.copy()

        # Fill the task graph with the tasks and decompose the edges that are not communicative
        self.update_graph()

        self.comm_graph: nx.Graph = nx.minimum_spanning_tree(self.comm_graph)

        self.leaf_nodes = [node for node, degree in self.comm_graph.degree() if degree == 1]

        # add the self loops again since the minimum spanning tree does not include them
        for state in start_positions.keys():
            self.comm_graph.add_edge(state, state)

        self.tokens = self.token_passing_algorithm(self.comm_graph)

        computeNewTaskGraph(self.task_graph, self.comm_graph, task_edges, start_position=start_positions)
        # self.print_tasks()    # Uncomment to print the tasks
        # self.plot_graph()     # Uncomment to plot the graphs


        # Wait for the controllers to be ready
        self.controller_timer = self.create_timer(0.5, self.wait_for_controller_callback)
        





    def update_graph(self):
        """Adds the tasks to the edges of the task graph."""
        for task_info in self.tasks.values():
            # Create the task
            task = self.create_task(task_info)
            # Add the task to the edge
            self.task_graph[task_info["EDGE"][0]][task_info["EDGE"][1]]["container"].add_tasks(task)


    def create_task(self, task_info) -> StlTask:
        """
        Creates a task based on the task information from the YAML file.
        
        Args:
            task_info (dict) : Task information from the YAML file.

        Returns:
            task (StlTask) : Task object.

        Raises:
            Exception : If the task type is not supported.

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
        # Create the predicate based on the type of the task
        if task_info["TYPE"] == "go_to_goal_predicate_2d":
            predicate = go_to_goal_predicate_2d(goal=np.array(task_info["CENTER"]), epsilon=task_info["EPSILON"], 
                                                agent=self.agents[task_info["INVOLVED_AGENTS"][0]])
        elif task_info["TYPE"] == "formation_predicate":
            predicate = formation_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], 
                                            agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]], relative_pos=np.array(task_info["CENTER"]))
        elif task_info["TYPE"] == "epsilon_position_closeness_predicate": 
            predicate = epsilon_position_closeness_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], 
                                                             agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]])
        else:
            raise Exception(f'Task type: {task_info["TYPE"]} is not supported')
        
        # Create the temporal operator
        if task_info["TEMP_OP"] == "AlwaysOperator":
            temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))
        elif task_info["TEMP_OP"] == "EventuallyOperator":
            temporal_operator = EventuallyOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))

        # Create the task
        task = StlTask(predicate=predicate, temporal_operator=temporal_operator)
        return task


    def plot_graph(self):
        """Plots the communication graph, initial task graph, and decomposed task graph."""
        fig, ax = plt.subplots(1, 3)
        self.draw_graph(ax[0], self.comm_graph, "Communication Graph")
        self.draw_graph(ax[1], self.initial_task_graph, "Initial Task Graph")
        self.draw_graph(ax[2], self.task_graph, "Decomposed Task Graph")
        plt.show()

    def draw_graph(self, ax, graph, title):
        """Draws the graph."""
        nx.draw(graph, with_labels=True, font_weight='bold', ax=ax)
        ax.set_title(title)


    def print_tasks(self):
        """Prints the tasks of the task graph."""
        for i,j,attr in self.task_graph.edges(data=True):
            tasks = attr["container"].task_list
            for task in tasks:
                print("-----------------------------------")
                print(f"EDGE: {list(task.edgeTuple)}")
                print(f"TYPE: {task.type}")
                print(f"CENTER: {task.center}")
                print(f"EPSILON: {task.epsilon}")
                print(f"TEMP_OP: {task.temporal_type}")
                print(f"INTERVAL: {task.time_interval.aslist}")
                print(f"INVOLVED_AGENTS: {task.contributing_agents}")
                print("-----------------------------------")


    
    
    def print_tokens(self, tokens:Dict[int,Ti]) :
        for unique_identifier,Ti in tokens.items() :
            print(f"Agent {unique_identifier} has tokens:")
            for j,token in Ti.items() :
                print(f"tau_{unique_identifier,j} = {token}")

    def update_tokens(self, unique_identifier : int,tokens_dictionary : Dict[int,Ti]) :
        
        Ti = tokens_dictionary[unique_identifier]
        
        for j in Ti.keys() :
            if len(Ti.keys()) == 1  : #edge leader
                Ti[j] = LeadershipToken.LEADER # set agent as edge leader
                
            else :   
                Tj = tokens_dictionary[j]
                if Tj[unique_identifier] == LeadershipToken.LEADER :
                    Ti[j] = LeadershipToken.FOLLOWER # set agent as edge follower
        
        count = 0
        for j,token in Ti.items():
            if token == LeadershipToken.UNDEFINED :
                count+= 1
                index = j
        if count == 1 : # there is only one undecided edge
            Ti[index] = LeadershipToken.LEADER # set agent as edge leader


    def any_undefined(self, Ti:Dict[int,LeadershipToken]) -> bool :
        for token in Ti.values():
            if token == LeadershipToken.UNDEFINED :
                return True
        return False


    def token_passing_algorithm(self, graph:nx.Graph):
        """
        implements token passing algorithm
        """
        # 0 undefined
        # 1 leader 
        # 2 follower
        tokens_dictionary = {}
        # parallelized version (the overhead from to little agents will not be worth it)
        
        if len(graph.nodes()) >= 10 :
            agents = list(graph.nodes())
            manager = mp.Manager()
            
            for agent in agents :
                neighbours = [unique_identifier for unique_identifier in graph.neighbors(agent) if unique_identifier!=agent] # eliminate self loops
                Ti = manager.dict({unique_identifier:LeadershipToken.UNDEFINED for unique_identifier in neighbours})
                tokens_dictionary[agent] = Ti
            
            tokens_dictionary = manager.dict(tokens_dictionary)
            diameter = net_diameter(graph)
        
            for round in range(int(np.ceil(diameter/2))+1) :
                with mp.Pool(6) as pool:
                    pool.starmap(self.update_tokens, [(agent, tokens_dictionary) for agent in agents])
            
            
            for Ti in tokens_dictionary.values() :
                if self.any_undefined(Ti) :
                    self.print_tokens(tokens_dictionary)
                    raise RuntimeError("Error: token passing algorithm did not converge")
    
            
        # serial version
        else :
            agents = list(graph.nodes())
            
            for agent in agents :
                Ti = {unique_identifier:LeadershipToken.UNDEFINED for unique_identifier in graph.neighbors(agent) if unique_identifier!=agent}
                tokens_dictionary[agent] = Ti
            
            diameter = net_diameter(graph)
            
            for round in range(int(np.ceil(diameter/2))+1) :
                for agent in agents :
                    self.update_tokens(agent,tokens_dictionary)
                    
            
            for Ti in tokens_dictionary.values() :
                if self.any_undefined(Ti) :
                    self.print_tokens(tokens_dictionary)
                    raise RuntimeError("Error: token passing algorithm did not converge")
                
            
        return tokens_dictionary



    # ==================== Publishers ====================

    def publish_leaf_nodes(self):
        """Publishes the leaf nodes to the topic leaf_nodes."""
        leaf_nodes_msg = LeafNodes()
        leaf_nodes_msg.list = self.leaf_nodes
        self.leaf_nodes_pub.publish(leaf_nodes_msg)

    def publish_tokens(self):
            """Publishes the tokens to the topic tokens."""
            LeadershipTokens_msg = LeaderShipTokens()
            for i,Ti in self.tokens.items():
                for j,token in Ti.items():
                    EdgeLeaderShip_msg = EdgeLeaderShip()
                    EdgeLeaderShip_msg.i = i
                    EdgeLeaderShip_msg.j = j
                    if token == LeadershipToken.LEADER:
                        EdgeLeaderShip_msg.leader = i
                    elif token == LeadershipToken.FOLLOWER:
                        EdgeLeaderShip_msg.leader = j
                    LeadershipTokens_msg.list.append(EdgeLeaderShip_msg)
            self.tokens_pub.publish(LeadershipTokens_msg)


    def publish_tasks(self):
        """Publishes the tasks to the topic tasks."""
        tasks: list[StlTask] = []
        for i,j,attr in self.task_graph.edges(data=True):
            tasks = attr["container"].task_list
            for task in tasks:
                self.total_tasks += 1
                task_message = TaskMsg()
                task_message.edge = list(task.edgeTuple)
                task_message.type = task.type                               
                task_message.center = task.center                           
                task_message.epsilon = task.epsilon                         
                task_message.temp_op = task.temporal_type
                task_message.interval = task.time_interval.aslist
                task_message.involved_agents = task.contributing_agents                             
                # Then publish the message
                self.task_pub.publish(task_message)


    def publish_numOfTask(self):
        """Publishes the number of tasks to the topic numOfTasks."""
        total = Int32()
        total.data = self.total_tasks
        self.numOfTasks_pub.publish(total)


    #  ==================== Callbacks ====================
    def controller_ready_callback(self, msg):
        agent = msg.data
        if agent not in self.ready_controllers:
            self.ready_controllers.append(agent)
        else:
            pass


    def wait_for_controller_callback(self):
        
        if all(agent in self.ready_controllers for agent in self.agents.keys()):
            self.controller_timer.cancel()
            self.publish_tokens()
            self.publish_leaf_nodes()
            self.publish_tasks()        
            self.publish_numOfTask()
        else:
            pass



def main(args=None):
    rclpy.init(args=args)
    manager = Manager()
    rclpy.spin(manager)
    manager.destroy_node()
    rclpy.shutdown() 


if __name__ == "__main__":
    main()

    