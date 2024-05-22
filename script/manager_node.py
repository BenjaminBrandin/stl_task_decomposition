#!/usr/bin/env python3
import yaml
import rclpy
from rclpy.node import Node
import numpy as np
import networkx as nx
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
from custom_msg.msg import task_msg
from decomposition_module import computeNewTaskGraph
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges
from builders import (Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, 
                      formation_predicate, epsilon_position_closeness_predicate, collision_avoidance_predicate)


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
        communication_radius: float = 4

        # setup publishers
        self.task_pub = self.create_publisher(task_msg, "tasks", 10)
        self.numOfTasks_pub = self.create_publisher(Int32, "numOfTasks", 10)

        # Load the initial states and the task from the yaml files
        with open("PATH/TO/YOUR/FILE.YAML") as file: # "/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/start_pos.yaml"
            self.start_pos = yaml.safe_load(file)  

        with open("PATH/TO/YOUR/FILE.YAML") as file: # "/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/tasks.yaml"
            self.tasks = yaml.safe_load(file)

        # Initial states of the robots and creating the robots
        start_positions: dict[int, np.ndarray] = {}
        for i, (state_key, state_value) in enumerate(self.start_pos.items(), start=1):
            self.agents[i] = Agent(id=i, initial_state=np.array(state_value))
            start_positions[i] = np.array(state_value)

        # Extracting the edges of the tasks
        task_edges = [tuple(task["EDGE"]) for task in self.tasks.values()]

        # Creating the graphs
        self.comm_graph = create_communication_graph_from_states(start_positions, communication_radius)  
        self.task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
        self.initial_task_graph = self.task_graph.copy()

        # Fill the task graph with the tasks and decompose the edges that are not communicative
        self.update_graph()
        computeNewTaskGraph(self.task_graph, self.comm_graph, task_edges, start_position=start_positions)
        
        # publish the tasks
        self.print_tasks()
        self.plot_graph()
        self.publish_numOfTask()
        self.publish_tasks()


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
            bool communicate 
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
        elif task_info["TYPE"] == "collision_avoidance_predicate":
            predicate = collision_avoidance_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], 
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
                self.total_tasks += 1
                print("-----------------------------------")
                print(f"EDGE: {list(task.edgeTuple)}")
                print(f"TYPE: {task.type}")
                print(f"CENTER: {task.center}")
                print(f"EPSILON: {task.epsilon}")
                print(f"TEMP_OP: {task.temporal_type}")
                print(f"INTERVAL: {task.time_interval.aslist}")
                print(f"INVOLVED_AGENTS: {task.contributing_agents}")
                print("-----------------------------------")
        rclpy.sleep(0.5)
        

    def publish_tasks(self):
        """Publishes the tasks to the task_pub."""
        tasks: list[StlTask] = []
        for i,j,attr in self.task_graph.edges(data=True):
            tasks = attr["container"].task_list
            for task in tasks:
                task_message = task_msg()
                task_message.edge = list(task.edgeTuple)
                task_message.type = task.type                               
                task_message.center = task.center                           
                task_message.epsilon = task.epsilon                         
                task_message.temp_op = task.temporal_type
                task_message.interval = task.time_interval.aslist
                task_message.involved_agents = task.contributing_agents                            

                # Then publish the message
                self.task_pub.publish(task_message)
                rclpy.sleep(0.5)

    def publish_numOfTask(self):
        """Publishes the number of tasks to the numOfTasks_pub."""
        flag = Int32()
        flag.data = self.total_tasks
        self.numOfTasks_pub.publish(flag)


def main(args=None):
    rclpy.init(args=args)
    manager = Manager()
    rclpy.spin(manager)
    manager.destroy_node()
    rclpy.shutdown() 

if __name__ == "__main__":
    main()

    