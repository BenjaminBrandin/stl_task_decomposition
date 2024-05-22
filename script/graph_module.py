"""
MIT License

Copyright (c) [2024] [Gregorio Marchesini]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import networkx as nx 
from builders import StlTask
from dataclasses import dataclass,field
from typing import Tuple, List, Dict, Union
UniqueIdentifier = int


@dataclass(unsafe_hash=True)
class EdgeTaskContainer:
    """
    Data class representing a graph edge, containing all the STL tasks defined over this edge.

    Attributes:
        _edge_tuple               (Tuple[UniqueIdentifier, UniqueIdentifier])  : Tuple of the edge.
        _weight                   (float)                                      : Weight of the edge.
        _task_list                (List[StlTask])                              : List of tasks of the edge.
        _involved_in_optimization (int)                                        : Flag that indicates if the edge is involved in the optimization or not.
    """

    _edge_tuple: Tuple[UniqueIdentifier,UniqueIdentifier]
    _weight: float = 1
    _task_list: List[StlTask] = field(default_factory=list)
    _involved_in_optimization: int = 0


    def __init__(self, 
                 edge_tuple: Tuple[UniqueIdentifier,UniqueIdentifier] = None, 
                 weight: float = None, 
                 task_list:List[StlTask] = None, 
                 involved_in_optimization: int = None):
        """
        Initializes the EdgeTaskContainer object.

        Args:
            edge_tuple               (Tuple[UniqueIdentifier, UniqueIdentifier])  : The tuple representing the edge.
            weight                   (float)                                      : The weight of the edge.
            task_list                (List[StlTask])                              : The list of tasks associated with the edge.
            involved_in_optimization (int)                                        : Flag indicating if the edge is involved in optimization or not.
        """

        self._edge_tuple               = edge_tuple               if edge_tuple is not None else (UniqueIdentifier(), UniqueIdentifier())
        self._weight                   = weight                   if weight is not None else 1
        self._task_list                = task_list                if task_list is not None else []
        self._involved_in_optimization = involved_in_optimization if involved_in_optimization is not None else 0


    @property
    def edge_tuple(self) -> Tuple[UniqueIdentifier, UniqueIdentifier]:
        """Returns the tuple of the edge."""
        return self._edge_tuple
    @property
    def weight(self) -> float:
        """Returns the weight of the edge."""
        return self._weight
    @property
    def task_list(self) -> List[StlTask]:
        """Returns the list of tasks of the edge."""
        return self._task_list
    @property
    def involved_in_optimization(self) -> int:
        """Returns the flag that indicates if the edge is involved in the optimization or not."""
        return self._involved_in_optimization
      

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Weight should be a positive number")
        for task in self.task_list:
            if not isinstance(task, StlTask):
                raise ValueError("The tasks list should contain only StlTask objects")
        
    def _add_single_task(self, input_task: StlTask) -> None:
        """ Set the tasks for the edge that has to be respected by the edge. Input is expected to be a list  """
       
        if not isinstance(input_task, StlTask):
            raise Exception("please enter a valid STL task object or a list of StlTask objects")
        else:
            if isinstance(input_task, StlTask):
                # set the source node pairs of this node
                nodei,nodej = self.edge_tuple

                if (not nodei in input_task.contributing_agents) or (not nodej in input_task.contributing_agents) :
                    raise Exception(f"the task {input_task} is not compatible with the edge {self.edge_tuple}. The contributing agents are {input_task.contributing_agents}")
                else:
                    self._task_list.append(input_task) # adding a single task
            
    # check task addition
    def add_tasks(self, tasks:Union[StlTask, List[StlTask]]):
        """ Add a single task or a list of tasks to the edge. """
        if isinstance(tasks, list): 
            for task in tasks:
                self._add_single_task(task)
        else: # single task case
            self._add_single_task(tasks)
    
    def cleanTasks(self)-> None :
        self.task_list      = []

    def flagOptimizationInvolvement(self) -> None:
        self._involved_in_optimization = 1
    

def create_task_graph_from_edges(edge_list:Union[List[EdgeTaskContainer], List[Tuple[UniqueIdentifier, UniqueIdentifier]]]) -> nx.Graph:
    """
    Create a task graph from a list of edges. 

    Args:
        edge_list (List[EdgeTaskContainer] or List[Tuple[UniqueIdentifier, UniqueIdentifier]]): List of edges to create the task graph.

    Returns:
        task_graph (nx.Graph): The task graph created from the edges.
    """
    task_graph = nx.Graph()
    for edge in edge_list:
        task_graph.add_edge(edge[0], edge[1], container=EdgeTaskContainer(edge_tuple=edge))
    
    return task_graph


def create_communication_graph_from_states(states: Dict[int, np.ndarray], communication_radius: float) -> nx.Graph:
    """
    Create a communication graph that is telling which states are communicating with each other.

    Args:
        states (Dict[int, np.ndarray]): A dictionary containing the initial states of the agents.
        communication_radius (float): The communication range of the agents.

    Returns:
        comm_graph (nx.Graph): The communication graph.

    Raises:
        Exception: If the edge is not compatible with the states.
    """    

    comm_graph = nx.Graph()
    comm_graph.add_nodes_from(states.keys())
    others = states.copy()

    # Add self-loops
    for state in states.keys():
        comm_graph.add_edge(state, state)

    for id_i,state_i in states.items() :
        others.pop(id_i)
        
        for id_j,state_j in others.items():
            distance = np.linalg.norm(state_i[:2]-state_j[:2])
            if distance <= communication_radius:
                comm_graph.add_edge(id_i,id_j)   
    
    return comm_graph