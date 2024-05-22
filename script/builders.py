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

#!/usr/bin/env python3
import itertools
import numpy as np
import casadi as ca
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, TypeVar, Optional, Union

UniqueIdentifier = int 
globalOptimizer = ca.Opti()

def first_word_before_underscore(string: str) -> str:
    """split a string by underscores and return the first element"""
    return string.split("_")[0]


def check_barrier_function_input_names(barrier_function: ca.Function)-> bool:
    """Check if the input names of the barrier function are in the form 'time' and 'state_i' where ''i'' is the agent ID."""
    for name in barrier_function.name_in():
        if not first_word_before_underscore(name) in ["state","time"]:
            return False
    return True    


def check_barrier_function_output_names(barrier_function: ca.Function)->bool:
    """Check if the output name of the barrier function is 'value'."""
    for name in barrier_function.name_out():
        if not first_word_before_underscore(name) == "value":
            return False
    return True


def is_time_state_present(barrier_function: ca.Function) -> bool:
    """Check if the time variable is present in the input names of the barrier function."""
    return "time" in barrier_function.name_in() 


def check_barrier_function_IO_names(barrier_function: ca.Function) -> bool:
    if not check_barrier_function_input_names(barrier_function) :
         raise ValueError("The input names for the predicate functons must be in the form 'state_i' where ''i'' is the agent ID and the output name must be 'value', got input nmaes " + str(function.name_in()) + " and output names " + str(function.name_out()) + " instead")
    
    elif not is_time_state_present(barrier_function) :
        raise ValueError("The time variable is not present in the input names of the barrier function. PLease make sure this is a function of time also (even if time could be not part of the barrier just put it as an input)")
    elif not check_barrier_function_output_names(barrier_function) :
        raise ValueError("The output name of the barrier function must be must be 'value'")
  

def check_predicate_function_input_names(predicate_function: ca.Function)-> bool:
    for name in predicate_function.name_in():
        if not first_word_before_underscore(name) in ["state"]:
            return False
    return True  


def check_predicate_function_output_names(predicate_function: ca.Function)->bool:
    for name in predicate_function.name_out():
        if not first_word_before_underscore(name) == "value":
            return False
    return True


def check_predicate_function_IO_names(predicate_function: ca.Function) -> bool:
    return check_predicate_function_input_names(predicate_function) and check_predicate_function_output_names(predicate_function)


def state_name_str(agent_id: UniqueIdentifier) -> str:
    """_summary_

    Args:
        agent_id (UniqueIdentifier): _description_

    Returns:
        _type_: _description_
    """    
    return f"state_{agent_id}"


def get_id_from_input_name(input_name: str) -> UniqueIdentifier:
    """Support function to get the id of the agents involvedin the satisfaction of this barrier function

    Args:
        input_names (list[str]): _description_

    Returns:
        list[UniqueIdentifier]: _description_
    """    
    if not isinstance(input_name,str) :
        raise ValueError("The input names must be a string")
    
 
    splitted_input_name = input_name.split("_")
    if 'state' in splitted_input_name :
        ids = int(splitted_input_name[1])
    else :
        raise RuntimeError("The input name must be in the form 'state_i' where ''i'' is the agent ID")
    
    return ids   


class Agent:
    """
    Stores information about an agent.

    Attributes:
        id              (int)           : The unique identifier of the agent.
        symbolic_state  (ca.MX)         : The symbolic representation of the agent's state.
        state           (np.ndarray)    : The initial state of the agent.
    """
    def __init__(self, id: int, initial_state: np.ndarray):
        """
        Initializes an Agent object.

        Args:
            id            (int)         : The unique identifier of the agent.
            initial_state (np.ndarray)  : The initial state of the agent.
        """
        self.id = id
        self.symbolic_state = ca.MX.sym(f'state_{id}', 2)
        self.state = initial_state


class PredicateFunction :
    """
    Definition class for predicate functions. This class stores information about a predicate function, including its name, function itself, 
    contributing agents, and an alternative function using the edge as input instead of the agents' state.

    Attributes:
            function_name        (srt)         : the name of the predicate function.
            function             (ca.Function) : the predicate function itself.
            function_edge        (ca.Function) : also the predicate function but it uses the edge as input instead of the state of the agents.
            stateSpaceDimension  (int)         : the dimension of the state space. Default is 2 because it is used for omnidirectional robots.
            computeApproximation (bool)        : if True, the class will create a hypercube approximation of the predicate function through optimization.
            sourceNode           (int)         : the source node of the edge.
            targetNode           (int)         : the target node of the edge.
            center               (List[float]) : the center of the predicate function.
            epsilon              (float)       : the threshold value used in predicate functions to determine the region of satisfaction.
    
    Note: 
        The alternative function is used in the task decomposition.
        When the function is set to None, the class will create a hypercube approximation of the predicate function through optimazation.
    """
    def __init__(self,
                 function_name: str,
                 function: ca.Function,
                 function_edge: ca.Function,
                 stateSpaceDimension: int = 2, 
                 computeApproximation = False,
                 sourceNode: int = None,
                 targetNode: int = None,
                 center: List[float] = None,
                 epsilon: float = None) -> None:
        
        """
        Initializes a PredicateFunction object. Depending on which predicate function is used, the input arguments will de different. 

        Args:
            function_name        (srt)         : the name of the predicate function.
            function             (ca.Function) : the predicate function itself.
            function_edge        (ca.Function) : also the predicate function but it uses the edge as input instead of the state of the agents.
            center               (List[float]) : the center of the predicate function.
            epsilon              (float)       : the threshold value used in predicate functions to determine the region of satisfaction.
        """
    
        self._function_name = function_name
        self._function      = function
        self._function_edge = function_edge
        self._dim           = stateSpaceDimension
        self._epsilon       = epsilon if epsilon is not None else 1
        self._center        = center if center is not None else [0.0, 0.0]


        if self._function != None:
            self._isParametric = 0
            self._centerVar    = None
            self._nuVar        = None
            self._etaVar       = None

            self._contributing_agents = [get_id_from_input_name(name) for name in self._function.name_in()]
            if len(self._contributing_agents) > 1:
                self._sourceNode, self._targetNode = self._contributing_agents[:2]
            else:
                self._sourceNode = self._targetNode = self._contributing_agents[0]
            self._edgeTuple = (self._sourceNode, self._targetNode)

        else:
            # Create hypercube parameters since the formula is parametric
            self._centerVar           = globalOptimizer.variable(self._dim, 1)    # casadi.Opti.variable() edge Casadi Variable for optimization
            self._nuVar               = globalOptimizer.variable(self._dim, 1)    # casadi.Opti.variable() nu variable (cuboid dimensions) for optimization
            self._etaVar              = ca.vertcat(self._centerVar, self._nuVar)  # casadi.Opti.variable() [centerVar, nuVar] (just concatenation)
            self._isParametric        = 1
            self._sourceNode          = sourceNode
            self._targetNode          = targetNode
            self._edgeTuple           = (sourceNode, targetNode)
            self._contributing_agents = [sourceNode, targetNode]
                
        self._isApproximated               = False
        self._optimalApproximationvertices = None
        self._optimalApproximationCenter   = None # center of the zero level set
        self._optimalApproximationNu       = None # nu vector of the cuboid containing the dimneioson of the cuboid
        
        if not self._isParametric and computeApproximation :
           self.computeBestCuboidApproximation()
           self._isApproximated = True
    
    @property
    def function(self) -> ca.Function:
        """Get the predicate function."""
        return self._function
    @property
    def function_name(self) -> str:
        """Get the name of the predicate function."""
        return self._function_name
    @property
    def function_edge(self) -> ca.Function:
        """Get the predicate function using the edge as input."""
        return self._function_edge
    @property
    def contributing_agents(self) -> List[UniqueIdentifier]:
        """Get the IDs of contributing agents."""
        return self._contributing_agents
    @property
    def epsilon(self) -> float:
        """Get the threshold value."""
        return self._epsilon
    @property
    def center(self) -> List[float]:
        """Get the center of the predicate function."""
        return self._center
    @property
    def stateSpaceDimension(self) -> int:
        """Get the dimension of the state space."""
        return self._dim
    @property
    def centerVar(self) -> ca.MX:
        """Get the center of the hypercube approximation."""
        return self._centerVar
    @property
    def nuVar(self) -> ca.MX:
        """Get the nu vector of the hypercube approximation."""
        return self._nuVar
    @property
    def sourceNode(self) -> int:
        """Get the source node of the edge."""
        return self._sourceNode
    @property
    def targetNode(self) -> int:
        """Get the target node of the edge."""
        return self._targetNode
    @property
    def edgeTuple(self) -> Tuple[int, int]:
        """Get the edge tuple."""
        return self._edgeTuple
    @property
    def hasUndefinedDirection(self) -> bool:
        """Check if the predicate has an undefined direction."""
        return ((self.sourceNode==None) or self.targetNode==None)
    @property
    def isParametric(self) -> bool:
        """Check if the predicate is parametric or not."""
        return self._isParametric
    @property
    def optimalApproximationCenter(self) -> ca.MX:
        """Get the center of the optimal approximation."""
        return self._optimalApproximationCenter
    @property
    def optimalApproximationNu(self) -> ca.MX:
        """Get the nu vector of the optimal approximation."""
        return self._optimalApproximationNu 


    def computeBestCuboidApproximation(self) :
        """
        Computes the best cuboid approximation of the predicate function. In the future this might be approximated by some more complex zonotopes if needed.

        Raises:
            Exception: If an error occurs while computing the cuboid approximation.

        """
        opti = ca.Opti()
        
        centerVarDummy = opti.variable(self._dim,1)
        nuVarDummy     = opti.variable(self._dim,1)
        
        cartesianProductSets         = [[-1,1],]*self._dim
        hypercubeVertices            = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
        parametericHypercubeVertices = centerVarDummy + hypercubeVertices*nuVarDummy/2                # Each column of the matrix represents one of the vertices
        
        # set dimensions to be positive
        opti.subject_to(nuVarDummy>=np.zeros((self._dim,1)))
        # for each vertex set inclusion in the original set
        numVertices = 2**self._dim
        for jj in range(numVertices) :
            vertex = parametericHypercubeVertices[:,jj]
            opti.subject_to(self._function(vertex)<=0) # each vertex must be contained in the zero level set of the cpnvex function
            
        # cost is the inverse of the volume
        volume = 1
        for kk in range(self._dim) :
            volume = volume * nuVarDummy[kk]
        
        cost = self._function(1/volume) # it must be convex
        
        # give good initial guess to avaoid initial high values
        opti.set_initial(centerVarDummy,self._centerGuess)
        opti.set_initial(nuVarDummy,np.ones(self._dim)*1.5) # just to not start with 0
        
        opti.minimize(cost)
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        opti.solver("ipopt", p_opts, s_opts)

        try  :
            opti.solve()
        except :
            raise Exception("Error occurred while computing cuboid approximation of predicate superlevel set. Plase verify the the predicate function is convex and that the starting guess is inside reasonable to find the minim of the function")
        
        self._optimalApproximationVertices = opti.value(parametericHypercubeVertices)
        self._optimalApproximationCenter   = opti.value(centerVarDummy) # center of the zero level set
        self._optimalApproximationNu       = opti.value(nuVarDummy) # nu vector of the cuboid containing the dimneioson of the cuboid
        
        
        if self._optimalApproximationCenter.ndim == 1:
            self._optimalApproximationCenter = self._optimalApproximationCenter[:,np.newaxis]
        if self._optimalApproximationNu.ndim == 1:
            self._optimalApproximationNu = self._optimalApproximationNu[:,np.newaxis]
    

    def computeCuboidApproximation(self) :
        """
        This function is called in case the original predicate needs to be replaced with an under approximation of it
        """
        if not self._isApproximated : # if the approximation was not computed before at the creation of the instance then create it now
            self.computeBestCuboidApproximation()
            self._isApproximated = True
        

    def hypercubeVertices(self,source:int,target:int) :
        """
        This function is called to obtain the vertices of the hypercube approximation of the predicate function. The vertices are computed based on the center and nu vector of the hypercube.

        Args:
            source (int): The source node of the edge.
            target (int): The target node of the edge.

        Returns:
            parametericHypercubeVertices : A matrix where each column represents one of the vertices of the hypercube.

        Raises:
            NotImplementedError: If the predicate has an undefined direction.
            NotImplementedError: If the given source and target do not match the edge of the predicate.
            NotImplementedError: If the formula is not parametric. 
        """
        
        if self.hasUndefinedDirection :
            raise NotImplementedError("predicate has undefined direction. Only if you define a target and source you can obtain the hypercube vertices")
        if self._isParametric :
            cartesianProductSets         = [[-1,1],]*self._dim
            hypercubeVertices            = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
            
            if (self.edgeTuple != (source,target)) and (self.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
                raise NotImplementedError("the given source\\target pair does not match the edge of the predicate")
            
            # if the direction of request matches the direction of the parametric predicate the use the normal center
            elif self.edgeTuple == (source,target) :
               # the requested direction is the orginal direction of the predicate
               parametericHypercubeVertices = self._centerVar + hypercubeVertices*self._nuVar/2    # Each column of the matrix represents one of the vertices
            
            else : # use the opposite direction center
                parametericHypercubeVertices = -1*self._centerVar + hypercubeVertices*self._nuVar/2   # Each column of the matrix represents one of the vertices
            
        else :
            raise NotImplementedError("formula is not parametric so you cannot quary any vertex through this function. If you wanted to get the best cuboid underappriximation, please check the property optimalApproximationVertices")
        
        return parametericHypercubeVertices
    

    def linearRepresentationHypercube(self,source:int,target:int) :
        """
        Returns linear representation of the parameteric function as Ax<=b. 
        
        Args:
            source (int): The source node of the edge.
            target (int): The target node of the edge.

        Returns:
            A : The matrix A of the linear representation.
            b : The vector b of the linear representation.
        
        Raises:
            NotImplementedError: If the predicate has an undefined direction or if the edge does not match the predicate.
            NotImplementedError: If the given source and target do not match the edge of the predicate.
            Exception: If the formula is not parametric and does not have an available linear approximation.
        """
        if self.hasUndefinedDirection :
            raise NotImplementedError("predicate has undefined direction. Only if you define a target and source you can obtain the hypercube vertices")
        
        if (self.edgeTuple != (source,target)) and (self.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
                raise NotImplementedError("the given source\\target pair dies not match the edge of the predicate")
        
        if self.isParametric : # parametric case
            
            # A(x-c) - b <= 0
            if self.edgeTuple == (source,target) :
                A  = np.vstack((np.eye(self.stateSpaceDimension),-np.eye(self.stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
                Ac = A@self._centerVar
                d  = ca.vertcat(self.nuVar/2,self.nuVar/2)
                b  = Ac+d
                return A,b
                
            else :
                A  = np.vstack((np.eye(self.stateSpaceDimension),-np.eye(self.stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
                Ac = A@(-self._centerVar)
                d  = ca.vertcat(self.nuVar/2,self.nuVar/2)
                b  = Ac+d
                return A,b
                         
        else :
            
            if self._isApproximated :
            
                # A(x-c) - b <= 0
                if self.edgeTuple == (source,target) :
                    A  = np.vstack((np.eye(self.stateSpaceDimension),-np.eye(self.stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
                    Ac = A@(self._optimalApproximationCenter)
                    d  = ca.vertcat(self.optimalApproximationNu/2,self.optimalApproximationNu/2)
                    b  = Ac+d
                    return A,b
                    
                else :
                    A  = np.vstack((np.eye(self.stateSpaceDimension),-np.eye(self.stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
                    Ac = A@(-self._optimalApproximationCenter)
                    d  = ca.vertcat(self.optimalApproximationNu/2,self.optimalApproximationNu/2)
                    b  = Ac+d
                    return A,b
            else :
                raise Exception("current formula is not paranetric and does not have an availble linear approximation. Please provide one by calling the appropriate method")
    
    

class BarrierFunction:
    """
    This is a class for convex barrier functions. It stores information about the barrier function, the associated alpha function, the switch function, and the time function."""
    
    def __init__(self,
                 function: ca.Function,
                 associated_alpha_function:ca.Function = None,
                 time_function:ca.Function = None,
                 switch_function:ca.Function = None,
                 name :str = None) -> None:
        """
        The initialization for a barrier function is a function b("state_1","state_2",...., "time")-> ["value].
        This type of structure it is checked within the class itself. The associated alpha function is is a scalar functon that can be used to 
        construct a barrier constraint in the form \dot{b}("state_1","state_2",...., "time") <= alpha("time")
        
        Args : 
            function                  (ca.Function) : the barrier function
            associated_alpha_function (ca.Function) : the associated alpha function
            switch_function           (ca.Function) : simple function of time that can be used to activate and deactivate the barrier. Namely the value is 1 if t<= time_of_remotion and 0 otherwise. Time is assumed to start from 0
        
        Example:
        >>> b = ca.Function("b",[state1,state2,time],[ca.log(1+ca.exp(-state1)) + ca.log(1+ca.exp(-state2))],["state_1","state_2","time"],["value"])
        >>> alpha = ca.Function("alpha",[dummy_scalar],[2*dummy_scalar])
        >>> barrier = BarrierFunction(b,alpha)

        Raises:
            TypeError: If the function is not a CasADi Function.
        """
    
        if not isinstance(function, ca.Function):
            raise TypeError("function must be a casadi.MX object") 
        
        check_barrier_function_IO_names(function) # will throw an exception if the wrong naming in input and output is given

        self._function :ca.Function = function
        self._switch_function       = switch_function
        self.check_that_is_scalar_function(function=associated_alpha_function) # throws an error if the given function is not valid one
        self._associated_alpha_function = associated_alpha_function
        self.check_that_is_scalar_function(function=time_function) # throws an error if the given function is not valid one
        self._time_function = time_function
        
        names = [name for name in function.name_in() if name != "time"] # remove the time
        self._contributing_agents = [get_id_from_input_name(name) for name in names] 
        self._gradient_function_wrt :dict[UniqueIdentifier,ca.Function] = {}
        
        self._partial_time_derivative :ca.Function = ca.Function()
        self._compute_local_gradient_functions() # computes local time derivatives and gradient functions wrt the state of each agent involved in the barrier function
        
        if name == None :
            self._name = self._function.name()
    

    @property
    def function(self):
        """Get the barrier function."""
        return self._function
    @property
    def name(self):
        """Get the name of the barrier function."""
        return self._name
    @property
    def associated_alpha_function(self):
        """Get the associated alpha function."""
        return self._associated_alpha_function
    @property
    def partial_time_derivative(self):
        """Get the partial derivative with respect to time."""
        return self._partial_time_derivative 
    @property
    def contributing_agents(self):
        """Get the IDs of contributing agents."""
        return self._contributing_agents
    @property
    def switch_function(self):
        """Get the switch function."""
        return self._switch_function
    @property
    def time_function(self):
        """Get the function of time."""
        return self._time_function
    

    def gradient_function_wrt_state_of_agent(self,agent_id) -> ca.Function:
        """
        Get the gradient function with respect to the state of a given agent.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            self._gradient_function_wrt (dict[UniqueIdentifier,ca.Function]): The gradient function with respect to the given agent.
        
        Raises:
            KeyError: If the gradient function for the specified agent is not stored.
        """
        try :
            return self._gradient_function_wrt[agent_id]
        except KeyError :
            raise KeyError("The gradient function with respect to agent " + str(agent_id) + " is not stored in this barrier function")
    

    # this function is applicable to general barriers
    def _compute_local_gradient_functions(self) -> None:
        """
        Store the local gradient of the barrier function with respect to the given agent id. 
        The stored gradient function takes as input the same names the barrier function
        """

        named_inputs : dict[str,ca.MX]  = {} # will contain the named inputs to the function
        input_names  : list[str]        = self._function.name_in()

        for input_name in input_names :
            variable = ca.MX.sym(input_name,self._function.size1_in(input_name)) # create a variable for each state
            named_inputs[input_name] = variable
        
        for input_name in input_names :

            if first_word_before_underscore(input_name) == "state":
                state_var = named_inputs[input_name]
                
                nabla_xi  = ca.jacobian(self._function.call( named_inputs)["value"] , state_var) # symbolic gradient computation
                state_id  = get_id_from_input_name(input_name)
                self._gradient_function_wrt[state_id] = ca.Function("nabla_x"+str(state_id),list(named_inputs.values()), [nabla_xi],input_names,["value"]) # casadi function for the gradient computation
            
            elif input_name == "time" :
                time_variable                 = named_inputs[input_name]
                partial_time_derivative       = ca.jacobian(self._function.call(named_inputs)["value"],time_variable) # symbolic gradient computation
                self._partial_time_derivative = ca.Function("local_gradient_function",list(named_inputs.values()), [partial_time_derivative],list(named_inputs.keys()),["value"]) # casadi function for the gradient computation
    
    
    def check_that_is_scalar_function(self,function:Optional[ca.Function]) -> None :
        """
        Check if the given function is a valid scalar function.

        Args:
            function (ca.Function): The function to check.

        Raises:
            TypeError: If the function is not a CasADi Function.
            ValueError: If the function is not a scalar function of one variable.
        """

        if function == None :
            pass
        else: 
            if not isinstance(function,ca.Function) :
                raise TypeError("The function must be a casadi function")
            if function.n_in() != 1 :
                raise ValueError("The  function must be a scalar function of one variable")
            if not  function.size1_in(0) == 1 :
                raise ValueError("The  function must be a scalar function of one variable")


class TimeInterval :
    """
    A class that represents a time interval with a start and end time.

    Attributes:
        _time_step: An integer that defines the time step used to switch time to sampling instants.
        Empty set is represented by a double a=None b = None.
    """
    _time_step = 1 

    def __init__(self, a: Union[float, int] = None, b: Union[float, int] = None) -> None:
        
        """
        Initializes a TimeInterval with a start time (a) and end time (b). If both a and b are None, the TimeInterval represents an empty set.

        Args:
            a: The start time of the interval. Must be a non-negative float or int.
            b: The end time of the interval. Must be a non-negative float or int and must be greater than or equal to a.

        Raises:
            ValueError: If a or b is not a non-negative float or int, or if b is less than a.
        """

        if any([a==None,b==None]) and (not all(([a==None,b==None]))) :
            raise ValueError("only empty set is allowed to have None Values in the interval")
        elif  any([a==None,b==None]) and (all(([a==None,b==None]))) : # empty set
           
            self._a = round(a/self._time_step)*self._time_step
            self._b = round(b/self._time_step)*self._time_step
            
        else :    
            # all the checks 
            if (not isinstance(a,float)) and  (not isinstance(a,int)) :
                raise ValueError("the input a must be a float or int")
            elif a<0 :
                raise ValueError("extremes of time interval must be positive")
            
            # all the checks 
            if (not isinstance(b,float)) and  (not isinstance(b,int)) :
                raise ValueError("the input b must be a float or int")
            elif b<0 :
                raise ValueError("extremes of time interval must be non negative")
            
            if a>b :
                raise ValueError("Time interval must be a couple of non decreasing time instants")
         
        self._a = a
        self._b = b
        
    @property
    def a(self):
        """Return the start time of the interval."""
        return self._a
    @property
    def b(self):
        """Return the end time of the interval."""
        return self._b
    @property
    def period(self):
        """Return the length of the time interval."""
        if self.is_empty():
            return None # empty set has measure None
        return self._b - self._a
    @property
    def aslist(self):
        """Return the time interval as a list."""
        return [self._a,self._b]
    
    def is_empty(self) -> bool:
        """Check if the time interval is empty."""
        if self.a is None and self.b is None:
            return True
        else:
            return False
        
    def is_singular(self) -> bool:
        """Check if the time interval is singular."""
        a, b = self.a, self.b
        if a == b:
            return True
        else:
            return False
    
    def __lt__(self,timeInt:TypeVar) -> TypeVar:
        """strict subset relations self included in timeInt ::: Self < timeInt """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.is_empty() and (not timeInt.is_empty()) :
            return True
        elif (not self.is_empty()) and timeInt.is_empty() :
            return False
        elif  self.is_empty() and timeInt.is_empty() : # empty set included itself
            return True
        else :
            if (a1<a2) and (b2<b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
    
    def __eq__(self,timeInt:TypeVar) -> bool:
        """ equality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
        
    def __ne__(self,timeInt:TypeVar) -> bool :
        """ inequality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
    def __le__(self,timeInt:TypeVar) -> TypeVar :
        """subset relations self included in timeInt  ::: Self < timeInt """
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.is_empty() and (not timeInt.is_empty()) :
            return True
        elif (not self.is_empty()) and timeInt.is_empty() :
            return False
        elif  self.is_empty() and timeInt.is_empty() : # empty set included itself
            return True
        else :
            if (a1<=a2) and (b2<=b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
        
    def __truediv__(self,timeInt:TypeVar) -> TypeVar :
        """interval Intersection"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        # the empty set is already in this cases since the empty set is included in any other set
        if timeInt<= self :
            return TimeInterval(a =timeInt.a, b = timeInt.b)
        elif self<= timeInt :
            return TimeInterval(a =self._a, b = self._b)
        else : # intersection case
            if (b1<a2) or (b2<a1) : # no intersection case
                return TimeInterval(a = None, b = None)
            elif (a1<=a2) and (b1<=b2) :
                return TimeInterval(a = a2, b = b1)
            else :
                return TimeInterval(a = a1, b = b2)



class TemporalOperator(ABC):
    """This is an abstract class for temporal operators. It is used to store the information about the temporal operators."""
    
    @property
    @abstractmethod
    def time_of_satisfaction(self) -> float:
        """The time when the formula is satisfied."""
        pass
    @property
    @abstractmethod
    def time_of_remotion(self) -> float:
        """The time when the formula is removed."""
        pass
    @property
    @abstractmethod
    def time_interval(self) -> 'TimeInterval':
        """The time interval in which the formula is satisfied."""
        pass  
    @property
    @abstractmethod
    def temporal_type(self) -> str:
        """The type of temporal operator."""
        pass

class AlwaysOperator(TemporalOperator):
    """
    Stores information about the Always operator.

    Attributes:
        _time_interval        (TimeInterval) : The time interval in which the formula is satisfied.
        _time_of_satisfaction (float)        : The time when the formula is satisfied.
        _time_of_remotion     (float)        : The time when the formula is removed.
        _temporal_type        (str)          : The type of temporal operator.
    """
    def __init__(self,time_interval:TimeInterval, temporal_type: str = "AlwaysOperator") -> None:

        self._time_interval         : TimeInterval = time_interval
        self._time_of_satisfaction  : float        = self._time_interval.a
        self._time_of_remotion      : float        = self._time_interval.b
        self._temporal_type         : str          = temporal_type
    
        """
        Initializes an AlwaysOperator object with the provided time interval.

        Args:
            time_interval (TimeInterval) : The time interval in which the formula is satisfied.
            temporal_type (str)          : The type of temporal operator.
        """

    @property
    def time_of_satisfaction(self) -> float:
        """The time when the formula is satisfied."""
        return self._time_of_satisfaction
    @property
    def time_of_remotion(self) -> float:
        """The time when the formula is removed."""
        return self._time_of_remotion
    @property
    def time_interval(self) -> TimeInterval:
        """The time interval in which the formula is satisfied."""
        return self._time_interval  
    @property
    def temporal_type(self) -> str:
        """The type of temporal operator."""
        return self._temporal_type
    
    
class EventuallyOperator(TemporalOperator):
    """
    Stores information about the Eventually operator.

    Attributes:
        _time_interval        (TimeInterval) : The time interval in which the formula is satisfied.
        _time_of_satisfaction (float)        : The time when the formula is satisfied.
        _time_of_remotion     (float)        : The time when the formula is removed.
        _temporal_type        (str)          : The type of temporal operator.
    """

    def __init__(self,time_interval:TimeInterval,time_of_satisfaction:float=None, temporal_type: str = "EventuallyOperator") -> None:
        """
        Initializes an EventuallyOperator object.

        Args:
            time_interval        (TimeInterval)    : The time interval in which the formula is satisfied.
            time_of_satisfaction (float, optional) : The time when the formula is satisfied. If not provided, it will be randomly generated within the time interval.
            temporal_type        (str)             : The type of temporal operator.

        Raises:
            ValueError: If the specified time of satisfaction is outside the time interval.
        """

        self._time_interval       : TimeInterval = time_interval
        self._time_of_satisfaction: float        = time_of_satisfaction
        self._time_of_remotion    : float        = self._time_interval.b
        self._temporal_type       : str          = temporal_type
        
        if time_of_satisfaction == None :
            self._time_of_satisfaction = time_interval.a + np.random.rand()*(time_interval.b- time_interval.a)
            
        elif time_of_satisfaction<time_interval.a or time_of_satisfaction>time_interval.b :
            raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{time_interval.a},{time_interval.b}]")
        
    @property
    def time_of_satisfaction(self) -> float:
        """The time when the formula is satisfied."""
        return self._time_of_satisfaction
    @property
    def time_of_remotion(self) -> float:
        """The time when the formula is removed."""
        return self._time_of_remotion
    @property
    def time_interval(self) -> TimeInterval:
        """The time interval in which the formula is satisfied."""
        return self._time_interval  
    @property
    def temporal_type(self) -> str:
        """The type of temporal operator."""
        return self._temporal_type


@dataclass(unsafe_hash=True)
class StlTask:
    """
    This class is used to store information about an STL task. It contains the predicate function and the temporal operator.

    Attributes:
        _predicate              (PredicateFunction) : The predicate function.
        _temporal_operator      (TemporalOperator)  : The temporal operator.

    Note:
        If the predicate is not provided then it will be set to None and will become parametric. 

    """
    def __init__(self, predicate: PredicateFunction = None, temporal_operator: TemporalOperator = None) -> None:
         
        self._predicate              = predicate         if predicate is not None else PredicateFunction
        self._temporal_operator      = temporal_operator if temporal_operator is not None else TemporalOperator

        """
        Initializes an STL task with the provided predicate function and temporal operator.

        Args:
            predicate         (PredicateFunction) : The predicate function.
            temporal_operator (TemporalOperator)  : The temporal operator.
        """

    
    @property
    def predicate(self) -> PredicateFunction:
        """Get the predicate."""
        return self._predicate
    @property
    def type(self) -> str:
        """Get the type of predicate function."""
        return self._predicate.function_name
    @property
    def epsilon(self) -> float:
        """Get the threshold value."""
        return self._predicate.epsilon
    @property
    def center(self) -> List[float]:
        """Get the center of the predicate function."""
        return self._predicate.center
    @property
    def temporal_operator(self) -> TemporalOperator:
        """Get the temporal operator."""
        return self._temporal_operator
    @property
    def temporal_type(self) -> str:
        """Get the type of temporal operator."""
        return self._temporal_operator.temporal_type
    @property
    def time_interval(self) -> TimeInterval:
        """Get the time interval of the temporal operator."""
        return self._temporal_operator.time_interval
    @property
    def predicate_function(self) -> ca.Function:
        """Get the predicate function."""
        return self._predicate.function
    @property
    def isParametric(self) -> bool:
        """Check if the predicate is parametric or not."""
        return self._predicate.isParametric
    @property
    def contributing_agents(self) -> List[int]:
        """Get the IDs of contributing agents."""
        return self._predicate.contributing_agents
    @property
    def centerVar(self) -> ca.MX:
        """Get the center variable."""
        return self._predicate.centerVar
    @property
    def nuVar(self) -> ca.MX:
        """Get the nu variable."""
        return self._predicate.nuVar
    @property
    def sourceNode(self) -> int:
        """Get the source node."""
        return self._predicate.contributing_agents[0]
    @property
    def targetNode(self) -> int:
        """Get the target node."""
        return self._predicate.contributing_agents[1]
    @property
    def edgeTuple(self) -> Tuple[int,int]:
        """Get the edge tuple."""
        return self._predicate.edgeTuple
    

    def getHypercubeVertices(self,sourceNode,targetNode:int) -> List[ca.MX]:
        """Computes vertices of hypercube as function of the centerVar and the dimension vector nuVar"""
        return self.predicate.hypercubeVertices(source=sourceNode,target=targetNode)
    
    def computeLinearHypercubeRepresentation(self,sourceNode:int,targetNode:int) -> Tuple[ca.MX,ca.MX] :
        """returns linear representation of the parameteric function as Ax<=b"""
        A,b = self._predicate.linearRepresentationHypercube(source=sourceNode,target=targetNode)
        return A,b
    
    def getConstraintFromInclusionOf(self,formula : 'StlTask') -> List[ca.MX]:
        """
        Returns inclusion constraints for "formula" inside the of the self formula instance.

        Args:
            formula (StlTask): The formula to include.

        Returns:
            constraints (List[ca.MX]): The inclusion constraints.

        Raises:
            ValueError: If the input formula is not an instance of the STL formula.
            NotImplementedError: If the formula you are trying to include is not part of the same edge as the current formula.
            NotImplementedError: If the formula you are trying to include is not parametric and the current formula is parametric.

        """
        if not isinstance(formula,StlTask) :
            raise ValueError("input formula must be an instance of STL formula")
        # two cases :
        # 1) parameteric vs parametric
        # 2) parameteric vs non-parameteric
        numVertices = 2**self.predicate.stateSpaceDimension
        constraints = []
        
        if formula.isParametric and self.isParametric :
            
            source,target = self.edgeTuple
            # get the represenations of both formulas with the right verse of the edge
            vertices      = formula.getHypercubeVertices(sourceNode=source,targetNode=target)
            A,b           = self.computeLinearHypercubeRepresentation(sourceNode=source,targetNode=target)
            constraints   = []
            for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
                constraints +=[ A@vertices[:,jj]-b<=np.zeros((self.predicate.stateSpaceDimension*2,1))]
                
        elif  formula.isParametric and (not self.isParametric) :
            source,target = self.edgeTuple # check the direction of definition
            
            if (formula.edgeTuple != self.edgeTuple) and (formula.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
                raise NotImplementedError("It seems that you are trying to make an inclusion between two formulas that are not part of the same edge. This is not support for now")
            
            vertices    = formula.getHypercubeVertices(sourceNode=source,targetNode=target)
            constraints = []
            
            if self._predicate._isApproximated : # in this case the predicate was approximated so you can do all of this with linear hyperplanes 
                A,b = self.computeLinearHypercubeRepresentation(sourceNode=source,targetNode=target)
                for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
                    constraints +=[ A@vertices[:,jj] - b<=0 ] 
            else :
                for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
                    constraints +=[ self._predicate.function_edge(vertices[:,jj])<=0 ] 
                
        elif (not formula.isParametric) and self.isParametric :  
            raise NotImplementedError("Trying to include a non parameteric formula inside a parameteric one. Not supported")
        
        return constraints
    

def go_to_goal_predicate_2d(goal:np.ndarray,epsilon :float, agent:Agent) -> PredicateFunction:

    """
    Helper function to create a go-to-goal predicate in the form ||position-goal|| <= epsilon.
    This predicate is useful to make an agent go to a certain goal position.

    Args:
        goal    (np.ndarray) : the coordinates of the goal position
        epsilon (float)      : the closeness value
        agent   (Agent)      : the agent that will go to the goal position

    Returns:
        PredicateFunction (PredicateFunction) : the predicate function

    Raises:
        ValueError: If the agent and goal have different position dimensions.
    """

    array_to_list = goal.tolist()  # Convert NumPy array to Python list
    goal_list     = [float(x) for x in array_to_list]  # Convert elements to integers
    
    if agent.symbolic_state.numel() != goal.size:
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(agent.symbolic_state.s) + " and " + str(goal.size) + 
                         "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    
    if len(goal.shape) <2 :
        goal = goal[:,np.newaxis]
        
    edge = ca.MX.sym(f"edge_{agent.id}{agent.id}", 2, 1)

    predicate_expression      = (epsilon**2 - ca.dot((agent.symbolic_state - goal),(agent.symbolic_state-goal))) # the scaling will make the time derivative smaller whe you built the barrier function
    predicate_expression_edge = (epsilon**2 - ca.dot((edge - goal),(edge-goal)))

    predicate_edge = ca.Function("go_to_goal_predicate_2d_edge",[edge], [predicate_expression_edge]) 
    predicate      = ca.Function("go_to_goal_predicate_2d",[agent.symbolic_state], 
                                                           [predicate_expression], 
                                                           ["state_"+str(agent.id)], 
                                                           ["value"]) # this defined an hyperplane function
    
    return PredicateFunction(function_name="go_to_goal_predicate_2d", function=predicate, function_edge=predicate_edge, center=goal_list, epsilon=epsilon)


def epsilon_position_closeness_predicate(epsilon:float, agent_i:Agent, agent_j:Agent) ->PredicateFunction: # Does not need to be called state_1 and state_2
    """
    Helper function to create a closeness relation predicate in the form ||position1-position2|| <= epsilon.
    This predicate is useful to dictate some closeness relation among two agents for example.
    
    Args:
        epsilon  (float) : the closeness value
        agent_i  (Agent) : the first agent
        agent_j  (Agent) : the second agent
    
    Returns:
        PredicateFunction (PredicateFunction) : the predicate function
    
    Raises:
        ValueError: If the two dynamical models have different position dimensions.
    """

    if agent_i.symbolic_state.shape != agent_j.symbolic_state.shape:
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(agent_i.symbolic_state.shape) + " and " + str(agent_j.symbolic_state.shape) + 
                         "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    edge = ca.MX.sym(f"edge_{agent_i.id}{agent_j.id}", 2, 1)

    predicate_expression      = (epsilon**2 - ca.sumsqr(agent_i.symbolic_state - agent_j.symbolic_state)) # the scaling will make the time derivative smaller whe you built the barrier function
    predicate_expression_edge = (epsilon**2 - ca.sumsqr(edge))

    predicate_edge = ca.Function("epsilon_position_closeness_edge",[edge],[predicate_expression_edge]) 
    predicate      = ca.Function("epsilon_position_closeness",[agent_i.symbolic_state, agent_j.symbolic_state], 
                                                              [predicate_expression],
                                                              ["state_"+str(agent_i.id),"state_"+str(agent_j.id)], 
                                                              ["value"]) # this defined an hyperplane function
    
    return PredicateFunction(function_name="epsilon_position_closeness_predicate", function=predicate, function_edge=predicate_edge, epsilon=epsilon)


def collision_avoidance_predicate(epsilon:float, agent_i:Agent, agent_j:Agent) ->PredicateFunction: 
    """
    Helper function to create a collision avoidance predicate in the form ||position1-position2|| <= epsilon.
    This predicate is useful th make sure that two agents do not collide.

    Args:
        epsilon  (float) : the closeness value
        agent_i  (Agent) : the first agent
        agent_j  (Agent) : the second agent

    Returns:
        PredicateFunction (PredicateFunction) : the predicate function
    
    Raises:
        ValueError: If the two dynamical models have different position dimensions.
    """


    if agent_i.symbolic_state.shape != agent_j.symbolic_state.shape:
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(agent_i.symbolic_state.shape) + " and " + str(agent_j.symbolic_state.shape) + 
                         "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    edge = ca.MX.sym(f"edge_{agent_i.id}{agent_j.id}", 2, 1)

    predicate_expression      = (ca.sumsqr(agent_i.symbolic_state - agent_j.symbolic_state) - epsilon**2) # the scaling will make the time derivative smaller whe you built the barrier function
    predicate_expression_edge = (ca.sumsqr(edge) - epsilon**2)

    predicate_edge = ca.Function("collision_avoidance_edge",[edge],[predicate_expression_edge]) 
    predicate      = ca.Function("collision_avoidance",[agent_i.symbolic_state, agent_j.symbolic_state], 
                                                       [predicate_expression],
                                                       ["state_"+str(agent_i.id),"state_"+str(agent_j.id)],
                                                       ["value"]) # this defined an hyperplane function
    
    return PredicateFunction(function_name="collision_avoidance_predicate", function=predicate, function_edge=predicate_edge, epsilon=epsilon)


def formation_predicate(epsilon:float, agent_i:Agent, agent_j:Agent, relative_pos:np.ndarray, direction_i_to_j:bool=True) ->PredicateFunction:
    """
    Helper function to create a closeness relation predicate witha  certain relative position vector. in the form ||position1-position2|| <= epsilon.
    This predicate is useful if you want to dictate a certain formation among two agents.
    
    Args:
        epsilon          (float)      : the closeness value
        agent_i          (Agent)      : the first agent
        agent_j          (Agent)      : the second agent
        relative_pos     (np.ndarray) : the relative position vector
    
    Returns:
        PredicateFunction (PredicateFunction) : the predicate function
    
    Raises:
        ValueError: If the two dynamical models have different position dimensions.
        ValueError: If the relative position vector has a different dimension than the agents' positions.
    """
    array_to_list     = relative_pos.tolist()  # Convert NumPy array to Python list
    relative_pos_list = [float(x) for x in array_to_list]  # Convert elements to integers

    if agent_i.symbolic_state.shape != agent_j.symbolic_state.shape:
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(agent_i.symbolic_state.shape) + " and " + str(agent_j.symbolic_state.shape) + 
                         "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    if relative_pos.shape[0] != agent_i.symbolic_state.shape[0]:
        raise ValueError("The relative position vector must have the same dimension as the position of the agents. Agents have position dimension " 
                         + str(agent_i.symbolic_state.shape) + " and the relative position vector has dimension " + str(relative_pos.shape) )
    
    edge = ca.MX.sym(f"edge_{agent_i.id}{agent_j.id}", 2, 1)
    if direction_i_to_j :
        predicate_expression =  (epsilon**2 - (agent_j.symbolic_state - agent_i.symbolic_state- relative_pos).T@(agent_j.symbolic_state-agent_i.symbolic_state - relative_pos)) # the scaling will make the time derivative smaller whe you built the barrier function   
    else :
        predicate_expression =  (epsilon**2 - (agent_i.symbolic_state - agent_j.symbolic_state- relative_pos).T@(agent_i.symbolic_state-agent_j.symbolic_state - relative_pos))

    predicate_expression_edge = (epsilon**2 - (edge-relative_pos).T@(edge-relative_pos))

    predicate_edge = ca.Function("position_epsilon_closeness_edge",[edge], [predicate_expression_edge]) 
    predicate      = ca.Function("position_epsilon_closeness",[agent_i.symbolic_state,agent_j.symbolic_state],
                                                              [predicate_expression],
                                                              ["state_"+str(agent_i.id),"state_"+str(agent_j.id)],
                                                              ["value"]) # this defined an hyperplane function

    return PredicateFunction(function_name="formation_predicate", function=predicate, function_edge=predicate_edge,center=relative_pos_list, epsilon=epsilon)

def conjunction_of_barriers(barrier_list:List[BarrierFunction], associated_alpha_function:ca.Function=None)-> BarrierFunction : 
    """
    Function to compute the conjunction of barrier functions. The function takes a variable number of barrier functions as input and returns a new barrier function that is the conjunction of the input barrier functions.
    
    Args:
        barrier_list (List[BarrierFunction]) : The list of barrier functions to be conjuncted.
        associated_alpha_function (ca.Function) : The associated alpha function to be used for barrier constraint construction.
    
    Returns:
        BarrierFunction (BarrierFunction) : The barrier function representing the conjunction of the input barrier functions.

    Raises:
        TypeError: If the input arguments are not BarrierFunction objects.
        ValueError: If the associated alpha function is not a casadi scalar function of one variable.
        
    Example :
    >>> b1 = BarrierFunction(...)
    >>> b2 = BarrierFunction(...)
    >>> b3 = BarrierFunction(...)
    >>> b4 = conjunction_of_barriers(b1,b2,b3)
    """
    
    # check if the input is a list of barrier functions
    for arg in barrier_list:
        if not isinstance(arg, BarrierFunction):
            raise TypeError("All the input arguments must be BarrierFunction objects.")
    
    # check that function is a scalar function
    if associated_alpha_function != None :
        if not isinstance(associated_alpha_function,ca.Function) :
            raise ValueError("The associated alpha function must be a casadi function")
        if associated_alpha_function.n_in() != 1 :
            raise ValueError("The associated alpha function must be a scalar function of one variable")
        if not  associated_alpha_function.size1_in(0) == 1 :
            raise ValueError("The associated alpha function must be a scalar function of one variable")
        
    
    contributing_agents = set()
    for barrier in barrier_list:
        contributing_agents.update(barrier.contributing_agents)
        
    # the barriers are all functions of some agents state and time. So we create such variables   minimum_approximation -> -1/eta log(sum -eta * barrier_i)
    inputs = {}
    for agent_id in contributing_agents :
        inputs["state_"+str(agent_id)] = ca.MX.sym("state_"+str(agent_id),barrier.function.size1_in("state_"+str(agent_id)))
    
    inputs["time"] = ca.MX.sym("time",1)
    dummy = ca.MX.sym("dummy",1)
    
    # each barrier function has an associates switch function. This function is equal to 1 if the barrier is active and 0 otherwise
    # we then use this function to create another switch. This switch is used to remove the barrier from the minimum
    # create at the conjunction. Namely, the switch sets the barrier to infinity when the barrier is not needed anymore.
    
    inf_switch = ca.Function("inf_switch",[dummy],[ca.if_else(dummy==1.,1.,10**20)]) 
    sum_switch = 0 # sum of the switches will be the final switch. namely, it will be zero when all the switches are zero
    
    min_list = []
    sum = 0
    eta = 10
    
    for barrier in barrier_list:
        # gather the inputs for this barrier
        barrier_inputs = {}
        switch : ca.Function = barrier.switch_function
        
        for agent_id in barrier.contributing_agents :
            barrier_inputs["state_"+str(agent_id)] = inputs["state_"+str(agent_id)] # gather the inputs for this barrier     
        barrier_inputs["time"] = inputs["time"] 

        sum_switch += switch(dummy)  
        sum        += ca.exp(-eta*barrier.function.call(barrier_inputs)["value"]) # sum of the exponentials
        
        min_list.append(barrier.function.call(barrier_inputs)["value"] + inf_switch(switch(barrier_inputs["time"])))
    
    
    #create the final switch function
    final_switch = ca.Function("final_switch",[dummy],[ca.if_else(sum_switch>=1,1,0)]) # the final switch is the switch that is zero when all the switches are zero
    
    # smooth min
    # conjunction_barrier = -1/eta*ca.log(sum) # the smooth minimum of the barriers
    # real min
    conjunction_barrier = ca.mmin(ca.vertcat(*min_list)) # the barrier function is the minimum of the barriers
    
    b = ca.Function("conjunction_barrier",list(inputs.values()),[conjunction_barrier],list(inputs.keys()),["value"]) # Now we can have conjunctions of formulas
    
    return BarrierFunction(function=b, associated_alpha_function=associated_alpha_function, switch_function=final_switch)


def create_barrier_from_task(task:StlTask, initial_conditions:List[Agent], alpha_function:ca.Function = None, t_init:float = 0) -> BarrierFunction:
    """
    Creates a barrier function from a given STLtask in the form of b(x,t) = mu(x) + gamma(t-t_init) 
    where mu(x) is the predicate and gamma(t) is a suitably defined time function 
    
    Args:
        task               (StlTask)     : the task for which the barrier function is to be created
        initial_conditions (List[Agent]) : the initial conditions of the agents
        alpha_function     (ca.Function) : the associated alpha function to be used for barrier constraint construction
        t_init             (float)       : the initial time of the barrier function in terms of the time to which the barrier should start
        
    Returns:
         (BarrierFunction) : the barrier function associated with the given task

    Raises:
        ValueError: If the initial conditions for the contributing agents are not complete.
        ValueError: If the initial condition for an agent has a different size than the state of the agent.
        ValueError: If the time of satisfaction of the task is less than the initial time of the barrier.
    """
    # get task specifics
    contributing_agents  = task.contributing_agents # list of contributing agents. In the case of this controller this is always going to be 2 agents : the self agent and another agent
    
    # check that all the agents are present
    if not all(any(agent_id == agent.id for agent in initial_conditions) for agent_id in contributing_agents):
        raise ValueError("The initial conditions for the contributing agents are not complete. Contributing agents are " + str(contributing_agents) + 
                         " and the initial conditions are given for " + str(initial_conditions.keys()))
    
    # # check that the sates sizes match
    for agent in initial_conditions:
        if not (task.predicate.function.size1_in("state_"+str(agent.id)) == agent.symbolic_state.shape[0]):
            raise ValueError("The initial condition for agent " + str(agent.id) + 
                             " has a different size than the state of the agent. The size of the state is " + str(task.predicate.function.size1_in("state_"+str(agent.id))) + 
                             " and the size of the initial condition is " + str(agent.symbolic_state.shape[0]))

    # determine the initial values of the barrier function 
    initial_inputs  = {state_name_str(agent.id): agent.state for agent in initial_conditions} # create a dictionary with the initial state of the agents
    symbolic_inputs = {}

    # STATE SYMBOLIC INPUT
    for agent in initial_conditions:
        symbolic_inputs["state_"+str(agent.id)] = ca.MX.sym("state_"+str(agent.id), task.predicate.function.size_in("state_"+str(agent.id))) # create a dictionary with the initial state of the agents
        
    predicate_fun = task.predicate.function
    predicate_initial_value  = predicate_fun.call(initial_inputs)["value"]
    symbolic_predicate_value = predicate_fun.call(symbolic_inputs)["value"]
        
    
    # gamma(t) function construction :
    # for always, eventually and Eventually always a linear decay function is what we need to create the barrier function.
    # on the other hand we need to add a sinusoidal part to guarantee the task satisfaction. The following intuition should help :
    # 1) An EventuallyAlways just says that an always should occur within a time in the eventually. So it is just a kinf od delayed always where the delay it is the time of satisfaction within the eventually
    # 2) An AlwaysEventually is a patrolling task. So basically this task says that at every time of the always, a certain predicate should evetually happen. This is equivalent to satisify the eventual;ly at regular periods within the always
    
    # we first construct the linear part the gamma function
    
    
    time_of_satisfaction = task.temporal_operator.time_of_satisfaction
    time_of_remotion     = task.temporal_operator.time_of_remotion

    # for each barrier we now create a time transient function :
    time_var = ca.MX.sym("time", 1) # create a symbolic variable for time 
    
    # now we adopt a scaling of the barrier so that the constraint is more feasible. 
    
    if (time_of_satisfaction-t_init) < 0 : # G_[0,b] case
        raise ValueError("The time of satisfaction of the task is less than the initial time of the barrier. This is not possible")
    
    if (time_of_satisfaction-t_init) == 0 : # G_[0,b] case
        gamma    = 0
    
    else :
        # normalization phase (this helps feasibility heuristically)
        
        normalization_scale = np.abs((time_of_satisfaction-t_init)/predicate_initial_value) # sets the value of the barrier equal to the tme at which I ned to satisfy the barrier. It basciaky sets the spatial and termpoal diemsnion to the same scale 
        predicate_fun = ca.Function("scaled_predicate",list(symbolic_inputs.values()),[task.predicate.function.call(symbolic_inputs)["value"]*normalization_scale],list(symbolic_inputs.keys()),["value"])
        predicate_initial_value =  predicate_fun.call(initial_inputs)["value"]
        symbolic_predicate_value = predicate_fun.call(symbolic_inputs)["value"]
        
        if (time_of_satisfaction-t_init) == 0 : # G_[0,b] case
            gamma    = 0
            
        else :  
            
            if predicate_initial_value <=0:
                gamma0 = - predicate_initial_value*1.2 # this gives you always a gamma function that is at 45 degrees decay. It is an heuristic
            elif predicate_initial_value >0:
                gamma0 =  - predicate_initial_value*0.8
            else:
                gamma0 = - predicate_initial_value
                
            a = gamma0/(time_of_satisfaction-t_init)**2
            b = -2*gamma0/(time_of_satisfaction-t_init)
            c = gamma0
            quadratic_decay = a*(time_var-t_init)**2 + b*(time_var-t_init) + c
            gamma    = ca.if_else(time_var <=time_of_satisfaction-t_init ,quadratic_decay,0) # piece wise linear function
            
    
    switch_function = ca.Function("switch_function",[time_var],[ca.if_else(time_var<= time_of_remotion,1.,0.)]) # create the gamma function
    gamma_fun       = ca.Function("gamma_function",[time_var],[gamma]) # create the gamma function
    
    
    
    # now add time to the inputs and find symbolic value of the barrier function
    symbolic_inputs["time"]  = time_var # add time to the inputs
    barrier_symbolic_output  = (symbolic_predicate_value + gamma) 
    
    # now we create the barrier function. We will need to create the input/output names and the symbolic inputs
    barrier_fun = ca.Function("barrierFunction",list(symbolic_inputs.values()),[barrier_symbolic_output],list(symbolic_inputs.keys()),["value"])
    
    return BarrierFunction(function = barrier_fun, 
                           associated_alpha_function = alpha_function,
                           time_function=gamma_fun,
                           switch_function=switch_function)