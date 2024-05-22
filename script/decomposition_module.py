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

import itertools
import numpy as np
import casadi as ca
import networkx as nx
from typing import Tuple , List, Dict
from graph_module import EdgeTaskContainer
from builders import StlTask, TimeInterval, PredicateFunction, globalOptimizer


def allNonParametric(listOfFormulas : List[StlTask]) -> bool:
    """
    Check if all formulas are non-parametric.
    
    Args:
        listOfFormulas (list[StlTask]) : list of formulas to be considered
    
    Returns:
        bool : True if all formulas are non-parametric, False otherwise

    Raises:
        ValueError : If the list of formulas is empty.
    """
    
    if len(listOfFormulas) == 0 :
        raise ValueError("Empty list not accepted")
    for formula in listOfFormulas :
        if formula.isParametric :
            return False
    return True 

def allParametric(listOfFormulas : List[StlTask]) -> bool:
    """
    Check if all formulas are parametric.
    
    Args:
        listOfFormulas (list[StlTask]) : list of formulas to be considered
    
    Returns:
        bool : True if all formulas are parametric, False otherwise

    Raises:
        ValueError : If the list of formulas is empty.
    """
    
    if len(listOfFormulas) == 0 :
        raise ValueError("Empty list not accepted")
    for formula in listOfFormulas :
        if not formula.isParametric :
            return False
    return True

def isThereMultipleIntersection(formulasList : List[StlTask]) -> bool:
    """
    Check if one always has more than one intersection with other always operators.
    
    Args:
        formulasList (list[StlTask]) : list of formulas to be considered.

    Returns:
        bool : True if there is more than one intersection, False otherwise.

    Raises:
        ValueError : If the list of formulas is empty.
    """
   
    if len(formulasList) == 0 :
        raise ValueError("Empty list not accepted")
    for i,formulai in enumerate(formulasList) :
        count = 0
        for j,formulaj  in enumerate(formulasList) :
            
            intervalsIntersection: TimeInterval = formulai.time_interval / formulaj.time_interval
            if (j!= i) and (not intervalsIntersection.is_empty()) : # if there is an intersection
                count += 1
                if count>1 :
                    return True
        
    return False

def haveSameTimeInterval(listOfFormulas : List[StlTask]) -> bool:
    """
    Check that all formulas might have the same time interval.
    
    Args:
        listOfFormulas (list[StlTask]) : list of formulas to be considered

    Returns:
        bool : True if all formulas have the same time interval, False otherwise

    Raises:
        ValueError : If the list of formulas is empty.
    """
    if len(listOfFormulas) == 0 :
        raise ValueError("Empty list not accepted")
    if len(listOfFormulas) == 1 :
        return True
    else :
        intervalChecker = listOfFormulas[0].time_interval
        for formula in listOfFormulas[1:]:
            if formula.time_interval!=intervalChecker :
                return False
        return True 

def computeTimeIntervalIntersection(formulas :List[StlTask]) -> TimeInterval :
    """
    Compute intersection of a list of formulas
    
    Args:
        formulas (list[StlTask]) : list of formulas to be considered

    Returns:
        intersection (TimeInterval) : intersection of the time intervals of the formulas
    """
    # time interval intersection can be computed in sequence 
    
    if len(formulas) <2 :
        raise ValueError("At least two formulas must be given to compute the interval intersection")
    
    intersection = formulas[0].time_interval / formulas[1].time_interval
    for formula in formulas[2:] :
        intersection = intersection / formula.time_interval
    
    return intersection

def computeVolume(vector):
    """
    Simply computes the product of elements in a vector.
    
    Args:
        vector (np.array) : vector to be considered

    Returns:
        prod (float) : product of the elements of the vector
    """
    prod = 1
    n,m = vector.shape
    length = max(n,m)
    for jj in range(length):
        prod = prod*vector[jj]
        
    return prod


def edgeSet(path:list, isCycle:bool=False) -> List[Tuple[int,int]]:
    """
    Given a list of nodes, it returns the edges of the path as (n,n+1)

    Args: 
        path    (list) : list of nodes in the path
        isCycle (bool) : if the path is a cycle or not
    
    Returns:
        edges (list[tuple]): list of tuples (n,n+1)
    """

    if not isCycle:
      edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    elif isCycle: # due to how networkx returns edges
      edges = [(path[i], path[i+1]) for i in range(-1, len(path)-1)]
        
    
    return edges


def pathMinkowskiSumVertices(listOfFormulas : List[StlTask], edgeList: List[Tuple[int,int]]) -> ca.MX  :
    """
    Given a set of edges with assigned formulas, it computes the vertices of the minkowski sum for the superlevel sets of the given formulas.
    Each formula is define alon one edge of the edgeList in the same sequence as the formulas are given. The edge information is used to 
    decide in which verse we should take the formulas if their direction of definiton is different from the direction of the path for example
    
    Args:
        listOfFormulas (list[StlTask])        : list of STL formulas to be considered.
        edgeList       (list[Tuple[int,int]]) : list of edges that contain the formulas in the same order as the formulas are given
    
    Results:
        minkowshySumVertices (cvxpy.Variable): returns a matrix where each column is a vertex of the Minkowski sum of the hypercubes defined by each edge in the list of edges.

    Raises:
        ValueError : If the list of formulas does not have the same length as the list of edges.
        ValueError : If the edge along the path is not matching the corresponding formula. Make sure that the edges order and the formulas order is correct
        Exception  : If the formula does not have an approximation available. Please replace formula with its approximation before calling this method

    """
    
    if len(listOfFormulas) != len(edgeList) :
        raise ValueError("list of formulas must have the same length of edges list")
    
    stateSpaceDimension   = listOfFormulas[0].predicate.stateSpaceDimension
    cartesianProductSets  = [[-1,1],]*stateSpaceDimension
    hypercubeVertices     = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
    
    center = 0 # center of the hypercube
    nuSum  = 0 # dimensions of the hypercube
    
    for formula,edgeTuple in zip(listOfFormulas,edgeList) :
    
        if formula.edgeTuple != edgeTuple and formula.edgeTuple!= (edgeTuple[1],edgeTuple[0]) :
            raise ValueError("edge along the path is not maching the corresponding formula. Make sure that the edges order and the formulas order is correct")
        
        elif formula.edgeTuple != edgeTuple : # case in which the directions are not matching
            if formula.isParametric :
                center = center - formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the formula is not parameteric 
                if formula.predicate._isApproximated :
                    center = center - formula.predicate.optimalApproximationCenter
                    nuSum  = nuSum  + formula.predicate.optimalApproximationNu
                else :
                    raise Exception("formula does not have an approximation available. PLease replace formula with its approximation before calling this method")
        else : # case in which the directions are matching
            if formula.isParametric :
                center = center + formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the formula is not parameteric 
                if formula.predicate._isApproximated :
                    center = center + formula.predicate.optimalApproximationCenter
                    nuSum  = nuSum  + formula.predicate.optimalApproximationNu
                else :
                    raise Exception("formula does not have an approximation available. PLease replace formula with its approximation before calling this method")
        
    minkowshySumVertices = center + hypercubeVertices*nuSum/2 # find final hypercube dimension
    
    return minkowshySumVertices


def computeOverloadingConstraints(edgeObj: EdgeTaskContainer) -> List[ca.Function]:
    """
    Given a list of formulas defined on a single edge, it computes the required inclusion constraints due to the conjunction of such formulas along the same edge
    
    Args:
        edgeObj (EdgeTaskContainer) : EdgeTaskContainer object containing the formulas to be considered.

    Returns:
        constraints (List[ca.Function]) : List of constraints. Each constraint is a function that represents the inclusion relationship between two formulas.

    """
    
    formulas = edgeObj.task_list
    # print(len(formulas))
    if len(formulas) == 1 :
        return  [] # with one single formula you don't need overloading constraints
    
    # Main assumption check
    alwaysFormulas :list[StlTask]= [formula for formula in formulas if formula.temporal_operator.temporal_type=="AlwaysOperator"]
    eventuallyFormulas : list[StlTask] = [formula for formula in formulas if formula.temporal_operator.temporal_type=="EventuallyOperator"]
    constraints = []

    sameTimeInterval = False
    if isThereMultipleIntersection(formulasList=alwaysFormulas) :
        if not haveSameTimeInterval(listOfFormulas=alwaysFormulas) : # if there are multiple intersections then they must have the same time interval
            raise NotImplementedError("Seems like there is at least a triple of always formulas that are intersecting in terms of time interval. This can be handled only if all the always operatrs have the same time interval. This is not the case for the formulas you inserted. Possible constrasting inclusion constraints would arise")
        else : # in the case they all have the same time interval divide in parameteric and not parametric and start the inclusion process 
            sameTimeInterval = True # flag so that computation does not have to be redone
            
    # PART 1 : resolve always vs always formulas intersections
    if sameTimeInterval: # all the always formulas have the same time interval
        sharedTimeInterval = alwaysFormulas[0].time_interval
        
        if allParametric(alwaysFormulas) :
            #in case the formulas are all parameteric then we can include them in a sequence of formulas.
            # Each formula includes its successor
            for index in range(len(alwaysFormulas)-1) :
                parentFormula = alwaysFormulas[index]
                childFormula  = alwaysFormulas[index+1]
                
                constraints  += parentFormula.getConstraintFromInclusionOf(childFormula) #linear inclusion among two parameteric formulas

            
        else : # case in which the always formulas are non intersecting
            parametricFormulas    = [formula for formula in alwaysFormulas if formula.isParametric]
            nonParametricFormulas = [formula for formula in alwaysFormulas if not formula.isParametric]
            if len(nonParametricFormulas)>1 :
                raise NotImplementedError("Seems like there is one edge that constains two always specifications with same time interval. This is not smart. Indeed two always formulas with same time interval require that the superleevl sets are intersecting. So just rewrite the formula using the interscetion of the two orginal superlevel sets as predicate superlevel set")
            
            else :
                alwaysFormulas = nonParametricFormulas + parametricFormulas # we reorder the always formulas such that the non parameteric formula is at the beginning. SO the sequence of inclusions is now correct
                for index in range(len(alwaysFormulas)-1) :
                    parentFormula = alwaysFormulas[index]
                    childFormula  = alwaysFormulas[index+1]
                    constraints  += parentFormula.getConstraintFromInclusionOf(childFormula)
        
        innerMostAlwaysFormula = alwaysFormulas[-1] # the always formula the is the last one to receive an inclusion constraint. All the other formulas include this one
            
        if len(eventuallyFormulas) :
            for eventuallyFormula in eventuallyFormulas :
                if eventuallyFormula.time_interval<=sharedTimeInterval: # always formulas have same
                    constraints  += innerMostAlwaysFormula.getConstraintFromInclusionOf(eventuallyFormula)
                    
    else : # in the case that they don't have the same time interval and the always operators intersections are fixed
        
        if len(alwaysFormulas) >=2 : 
            combinations   = itertools.combinations(range(len(alwaysFormulas)),2) # all possible combinations of always formulas

            # check which combinations have intersections of time intervals and act accordingly
            for combo in combinations:
                formulaI:StlTask          = formulas[combo[0]]
                formulaJ:StlTask          = formulas[combo[1]]
                intersection:TimeInterval = formulaI.time_interval / formulaJ.time_interval
                
                if not intersection.is_empty() :
                    if formulaJ.isParametric and not(formulaI.isParametric) : # parametric vs non-parameteric formula
                        constraints  += formulaI.getConstraintFromInclusionOf(formulaJ)
                        
                    elif formulaI.isParametric and not(formulaJ.isParametric) :  
                        constraints  += formulaJ.getConstraintFromInclusionOf(formulaI)
                    
                    elif formulaI.isParametric and formulaJ.isParametric : # if both parameteric include the one with shortest time into the one with longest time (this is our design choice)
                        
                        if formulaI.time_interval.period  >=  formulaJ.time_interval.period: # then include the formula J in I
                            
                            constraints  += formulaI.getConstraintFromInclusionOf(formulaJ)
                        else :
                            constraints  += formulaJ.getConstraintFromInclusionOf(formulaI)
    
                    else : # if both formulas are non-parameteric then we assume that the disgn of the initial specification was such that it was possible to satisfy them in the first place
                        pass    
        
        
        if len(eventuallyFormulas) :
            for eventuallyFormula in eventuallyFormulas :
                for alwaysFormula in alwaysFormulas :

                    if eventuallyFormula.time_interval<=alwaysFormula.time_interval: # if the time interval of the always formulas includes the time interval of the eventually formula
                        if alwaysFormula.isParametric :
                            constraints  += alwaysFormula.getConstraintFromInclusionOf(eventuallyFormula)
                            break # since the always are only intersecting, an eventually can be strictly included only in of of them. 
                                # At most, there is another intersection with another always, but in this case there is no need for forcing the intersection.
                                # In addition, if all the always have the same time interval, the we would eliminate redundant constraints by only putting the 
                                # inclusion on one always since the always are also aready included into each other
                         
    return constraints


def createCycleClosureConstraint(cycleEdgeObjs : List[EdgeTaskContainer],cycleEdges : List[Tuple[int,int]]) -> List[ca.MX] :
    """
    Creates the cycle closure constraints for a given cycle.

    This function takes a list of EdgeTaskContainer objects and corresponding edges that form a cycle. 
    It computes the possible combinations of formulas that can close the cycle and returns the constraints that represent the inclusion of the negative Minkowski sum in the original Minkowski sum.

    Args:
        cycleEdgeObjs (List[EdgeTaskContainer]) : A list of EdgeTaskContainer objects defined over the cycle.
        cycleEdges    (List[Tuple[int,int]])    : A list of edges that form the cycle. Each edge corresponds to an EdgeTaskContainer in 'cycleEdgeObjs'.

    Returns:
        constraints (List[ca.MX]): A list of constraints representing the inclusion of the negative Minkowski sum in the original Minkowski sum.

    """

    constraints = []
    if len(cycleEdgeObjs)==0:
        return []
    
    cycleFormulas        : list[list[StlTask]] = [edge.task_list for edge in cycleEdgeObjs]
    possibleCombinations : list[list[StlTask]] = itertools.product(*cycleFormulas) # all possible combinations of formulas closing a cycle 
    stateSpaceDim        = cycleFormulas[0][0].predicate.stateSpaceDimension
    
    for combination in possibleCombinations:
        # each combination represent a combinaton of formulas that can possibly close the cycle
        alwaysFormulasInCombination     = [formula for formula in combination if formula.temporal_operator == "AlwaysOperator"]
        eventuallyFormulasInCombination = [formula for formula in combination if formula.temporal_operator == "EventuallyOperator"]
        
        if not allNonParametric(combination):
            
            # Case 1: all always formulas 
            if len(combination) == len(alwaysFormulasInCombination) and len(alwaysFormulasInCombination)!=0:
                intervalIntersection : TimeInterval = computeTimeIntervalIntersection(alwaysFormulasInCombination)
                
                if not intervalIntersection.is_empty(): # if there is an intersection then add the constraints 
                    # need to provide aprimate predicate first
                    # this function will change the predicate in case the predicate in non parametric it will computed its cuboid approximation and replace the orginal predicate
                    for formula in combination:
                        if not formula.isParametric :
                            formula.predicate.computeCuboidApproximation() # you need to make the replacement of the originaal predicate with its cuboid under-approximation
                    constraints += computeMinkowskiInclusionConstraintsForCycle(cycleFormulas = combination,stateSpaceDimension = stateSpaceDim,cycleEdges = cycleEdges)
            
            # Case 2:  case in which there are also eventually formulas 
            else :
                if len(eventuallyFormulasInCombination)  ==1 :
                    eventuallyFormula    = eventuallyFormulasInCombination[0]
                    intervalIntersection = computeTimeIntervalIntersection(alwaysFormulasInCombination)
                    # check if the eventually is included in intervalIntersection
                    if eventuallyFormula.time_interval <= intervalIntersection :
                        for formula in combination:
                            if not formula.isParametric :
                                formula.predicate.computeCuboidApproximation() # you need to make the replacement of the original predicate with its cuboid under-approximation
        
                        constraints +=  computeMinkowskiInclusionConstraintsForCycle(cycleFormulas = combination,stateSpaceDimension = stateSpaceDim,cycleEdges = cycleEdges)
                        
                elif len(eventuallyFormulasInCombination)  >1 :    
                    # now there is more than one eventually formula so try to check if all the eventually formulas have a single instant as time inteval : like [t,t] in the time interval
                
                    requiresConstraint = True
                    intervalChecker = eventuallyFormulasInCombination[0].time_interval 
                    for eventuallyFormula in eventuallyFormulasInCombination[1:]:  
                        if not eventuallyFormula.time_interval.is_singular():    
                            requiresConstraint = False
                            break # in case even only one eventually formula has non-singular time interval (singular time interval means like [t,t]) then you don't need to add constraints
                        else :
                            if intervalChecker != eventuallyFormula.time_interval :  # in case the singular time interval is not the same as all the others then you don't need a constraint
                                requiresConstraint = False
                                break
                
                    if requiresConstraint :
                        for formula in combination:
                            if not formula.isParametric :
                                formula.predicate.computeCuboidApproximation() # you need to make the replacement of the originaal predicate with its cuboid under-approximation
        
                        constraints +=  computeMinkowskiInclusionConstraintsForCycle(cycleFormulas = combination,stateSpaceDimension = stateSpaceDim,cycleEdges = cycleEdges)
                else :
                    pass
              
    return constraints


def computeMinkowskiInclusionConstraintsForCycle(cycleFormulas:List[StlTask], cycleEdges:List[Tuple[int,int]],stateSpaceDimension : int ) -> List[ca.MX]:
    """
    Computes the Minkowski inclusion constraints for a given cycle.

    This function takes a list of STL formulas and corresponding edges that form a cycle, along with the state space dimension. 
    It computes the Minkowski sum of the formulas along the cycle and returns the constraints that represent the inclusion of the negative Minkowski sum in the original Minkowski sum.

    Args:
        cycleFormulas       (List[StlTask])        : A list of STL formulas defined over the cycle.
        cycleEdges          (List[Tuple[int,int]]) : A list of edges that form the cycle. Each edge corresponds to a formula in 'cycleFormulas'.
        stateSpaceDimension (int)                  : The dimension of the state space.

    Returns:
        constraints (List[ca.MX]): A list of constraints representing the inclusion of the negative Minkowski sum in the original Minkowski sum.

    Note:
        The function assumes that the minimum cycle has at least 3 formulas (i.e., it forms a triangle or more complex shape). If the cycle has fewer formulas, the function may not behave as expected.
    """
    
    
    constraints = []
    numVertices = 2**stateSpaceDimension
    
            
    # in case we select the mid point of the cycle (which must have at least 3 formulas because a minimum cycle is a triangle) 
    midIndex           = int(len(cycleFormulas)/2)
    leftFormulas       = cycleFormulas[0:midIndex]
    rightFormulas      = cycleFormulas[midIndex:]
    leftPathEdges      = cycleEdges[0:midIndex]
    rightPathEdges     = cycleEdges[midIndex:]
        
    # compute respective Minkowski sum
    leftVertices     = pathMinkowskiSumVertices(leftFormulas,leftPathEdges)
    A,b              = minkowskiSumLinearRepresentation(rightFormulas,rightPathEdges)
        
    # now we are ready to add the inclusion constraint
    for jj in range(numVertices) :
        constraints += [A@(-leftVertices[:,jj])-b<=np.zeros((2*stateSpaceDimension))] # note the minus sign it is because you need the negative MinkowskySum to be included in the original minkowsky sum
        

    return constraints


def minkowskiSumLinearRepresentation(listOfFormulas : List[StlTask], edgeList:List[Tuple[int,int]]) -> Tuple[ca.MX,ca.MX] :
    """
    Given a list of formulas and the corresponding edges directions of path over which the formulas are defined, we  
    compute the matrix A and b representing the minkowski sum of the formulas along the given path 
    such that the minkowsky sum can be represneted as the linear inequality Ax<=b.

    Args:
        listOfFormulas (list[StlTask]) : list of STL formulas
        edgeList       (list[Tuple[int,int]]) : list of edges that contain the formulas in the same order as the formulas are given

    Returns:
        A (ca.MX) : matrix A of the linear inequality Ax<=b
        b (ca.MX) : vector b of the linear inequality Ax<=b

    Raises:
        ValueError : If the edge along the path is not matching the corresponding formula. Make sure that the edges order and the formulas order is correct
        Exception  : If the formula does not have an approximation available. Please replace formula with its approximation before calling this method
    
    """
    
    stateSpaceDimension  = listOfFormulas[0].predicate.stateSpaceDimension
    center     = 0 # center of the hypercube
    nuSum      = 0 # dimensions of the hypercube
    

    for formula,edgeTuple in zip(listOfFormulas,edgeList) :
    
        if formula.edgeTuple != edgeTuple and formula.edgeTuple!= (edgeTuple[1],edgeTuple[0]) :
            raise ValueError("edge along the path is not maching the corresponding formula. Make sure that the edges order and the formulas order is correct")
        
        elif formula.edgeTuple != edgeTuple : # case in which the directions are not matching
            if formula.isParametric :
                center = center - formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the formula is not parameteric 
                if formula.predicate._isApproximated :
                    center = center - formula.predicate.optimalApproximationCenter
                    nuSum  = nuSum  + formula.predicate.optimalApproximationNu
                else :
                    raise Exception("formula does not have an approximation available. PLease replace formula with its approximation before calling this method")
        else : # case in which the directions are matching
            if formula.isParametric :
                center = center + formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the formula is not parameteric 
                if formula.predicate._isApproximated :
                    center = center + formula.predicate.optimalApproximationCenter
                    nuSum  = nuSum  + formula.predicate.optimalApproximationNu
                else :
                    raise Exception("formula does not have an approximation available. PLease replace formula with its approximation before calling this method")
        
    A  = np.vstack((np.eye(stateSpaceDimension),-np.eye(stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
    Ac = A@center
    d  = ca.vertcat(nuSum/2,nuSum/2)
    b  = Ac+d
    return A,b


def computeNewTaskGraph(task_graph:nx.Graph, comm_graph:nx.Graph, task_edges:List[tuple], start_position: Dict[int, np.ndarray], problemDimension = 2, maxInterRobotDistance = 3)-> List[StlTask]: 
    """
    Solves the task decomposition completely
    
    Args:
        task_graph            (nx.Graph)              : The task graph.
        comm_graph            (nx.Graph)              : The communication graph.
        task_edges            (List[tuple])           : List of edges in the task graph.
        start_position        (Dict[int, np.ndarray]) : Start position of the agents.
        problemDimension      (int)                   : Dimension of the problem. Default is 2 because we are working with omnidirectional robots on a 2D plane.
        maxInterRobotDistance (float)                 : Maximum distance constraint for each hypercube vertex.

    """
    
    numberOfVerticesHypercube = 2**problemDimension
    
    pathsList                   : list[list[int]] = []
    pathConstraints             : list[ca.MX]     = []
    overloadingConstraints      : list[ca.MX]     = []
    cyclesConstraints           : list[ca.MX]     = []
    maxCommunicationConstraints : list[ca.MX]     = []
    positiveNuConstraint        : list[ca.MX]     = []

    decompositionOccurred = False
    decompositionSolved   = []
    edges_to_remove       = []

    for edge in task_edges:

        edge_container  :EdgeTaskContainer = task_graph[edge[0]][edge[1]]["container"] 
        has_tasks       :bool              = len(edge_container.task_list) > 0
        isCommunicating :bool              = comm_graph.has_edge(edge[0],edge[1])
    

        if (not isCommunicating) and (has_tasks): 
            decompositionOccurred = True
            edges_to_remove.append(tuple(edge))

            # retrive all the formulas to be decomposed
            formulasToBeDecomposed: list[StlTask] = [task for task in edge_container.task_list]

            # path finding and grouping nodes
            path = nx.shortest_path(comm_graph,source=edge[0],target=edge[1]) # path of agents from start to end
            pathsList.append(path) # save sources list for later plotting
            # for each formula to be decomposed we will have n subformulas with n being the length of the path we select.

            for formula in formulasToBeDecomposed : # add a new set of formulas for each edge
                edgeSubformulas : list[StlTask] = [] # list of subformulas associate to one orginal formula. you have as many subformulas as number of edges
                
                originalTemporalOperator       = formula.temporal_operator                      # get time interval of the orginal operator
                originalPredicate              = formula.predicate                              # get the predicate function
                originalEdgeTuple              = tuple(formula.contributing_agents)             # get the edge tuple
                
                if originalEdgeTuple == (edge[0],edge[1]) : #case the direction is correct
                    edgesThroughPath = edgeSet(path=path) # find edges along the path
                else :
                    edgesThroughPath =  edgeSet(path=path[::-1]) # we reverse the path. This is to follow the specification direction
                
                
                for sourceNode,targetNode in edgesThroughPath:
                    
                    # create a new parameteric subformula object
                    subformula = StlTask(predicate=PredicateFunction(function_name=originalPredicate.function_name, function=None, function_edge=originalPredicate.function_edge, sourceNode=sourceNode, targetNode=targetNode, center=originalPredicate.center, epsilon=originalPredicate.epsilon), temporal_operator=originalTemporalOperator)
                    
                    # warm start of the variables involved in the optimization TODO: check if you have a better warm start base on the specification you have. Maybe some more intelligen heuristic
                    globalOptimizer.set_initial(subformula.centerVar, start_position[targetNode]-start_position[sourceNode]) # target - source: Following the original
                    globalOptimizer.set_initial(subformula.nuVar, np.ones(problemDimension)*4)

                    # add subformulas to the current path
                    edgeSubformulas.append(subformula)
                    subformulaVertices = subformula.getHypercubeVertices(sourceNode=sourceNode,targetNode=targetNode)
                    
                    # set positivity of dimensions vector nu
                    positiveNuConstraint.append(-np.eye(problemDimension)@subformula.nuVar<=np.zeros((problemDimension,1))) # constraint on positivity of the dimension variable
                    
                    if task_graph.has_edge(sourceNode,targetNode):
                        task_graph[sourceNode][targetNode]["container"].add_tasks(subformula) # add the subformula to the edge container
                    else:
                        task_graph.add_edge(sourceNode,targetNode,container=EdgeTaskContainer(edge_tuple=(sourceNode,targetNode)))
                        task_graph[sourceNode][targetNode]["container"].add_tasks(subformula) # add the subformula to the edge container
                                            

                    # Set maximum distance constraint for each hypercube vertex
                    if maxInterRobotDistance != None :
                        for jj in range(numberOfVerticesHypercube) : 
                            maxCommunicationConstraints.append(ca.norm_2(subformulaVertices[:,jj])<=maxInterRobotDistance)     
                   
                
                # now set that the final i sum has to stay inside the original predicate
                minowkySumVertices  = pathMinkowskiSumVertices(edgeSubformulas,  edgesThroughPath)  # return the symbolic vertices f the hypercube to define the constraints
                for jj in range(numberOfVerticesHypercube):
                    pathConstraints.append(originalPredicate.function_edge(minowkySumVertices[:,jj])<=0) # for each vertex of the minkowski sum ensure they are inside the original predicate superlevel-set
            
            decompositionSolved.append((path,edgeSubformulas))    

            # mark all the used edges for the optimization
            edgesThroughPath = edgeSet(path=path) # find edges along the path
            # flag the edges applied for the optimization 
            for sourceNode,targetNode in edgesThroughPath :
                task_graph[sourceNode][targetNode]["container"].flagOptimizationInvolvement()

    # Remove the edges that have been decomposed
    task_graph.remove_edges_from(edges_to_remove)
    
    if decompositionOccurred :
        # adding cycles constraints to the optimization problem
        cycles :list[list[int]] = sorted(nx.simple_cycles(task_graph))
        cycles = [cycle for cycle in cycles if len(cycle)>1] # eliminate self loopscycles)
        for omega in cycles :
            cycleEdges    = edgeSet(omega,isCycle=True)
            cycleEdgesObj :list[list[EdgeTaskContainer]] = [task_graph[i][j]["container"] for i,j in cycleEdges ] 
            cyclesConstraints += createCycleClosureConstraint(cycleEdgeObjs=cycleEdgesObj,cycleEdges=cycleEdges)
            
        # now we compute the overloading constraints on a single objects
        # one line of overloading constraints
        optimisedEdges = [(i,j,attr["container"]) for i,j,attr in task_graph.edges(data=True) if attr["container"].involved_in_optimization] #list of edges that contain formulas that were created in the task decomposition 
        for i,j,attr in optimisedEdges:
            overloadingConstraints += computeOverloadingConstraints(attr)
        
                
        # #########################################################################################################
        # # OPTIMIZATION
        # #########################################################################################################

        cost = 0 # compute cost for parameetric formulas
        for i,j,attr in optimisedEdges:
            for formula in attr.task_list:
                if formula.isParametric :
                    cost = cost + 1/computeVolume(formula.nuVar)
            
            
        constraints = [*maxCommunicationConstraints,*positiveNuConstraint,*pathConstraints,*cyclesConstraints,*overloadingConstraints]
        globalOptimizer.subject_to(constraints) # Maximum Distance of a constraint


        options = {"ipopt": {"print_level": 0}}
        globalOptimizer.solver("ipopt", options)
        solution = globalOptimizer.solve()


        ###########################################################################################################
        # PRINT SOLUTION
        #########################################################################################################

        newFormulasCount = 0
        for i,j,attr in optimisedEdges:
            newFormulasCount += len([formula for formula in attr.task_list if formula.isParametric])
        
        print("-----------------------------------------")   
        print("Internal Report")   
        print("-----------------------------------------")   
        print(f"Total number of formulas created : {newFormulasCount}")   
        print("---------Found Solution------------------") 
        for path,formulas in decompositionSolved :   
            print("path: ",path)
            for formula in formulas:
                print("====================================") 
                print(f"EDGE: {formula.edgeTuple}")
                print(f"TYPE: {formula.type}")
                print(f"CENTER: {formula.center}")
                print(f"EPSILON: {formula.epsilon}")
                print(f"TEMP_OP: {formula.temporal_type}")
                print(f"INTERVAL: {formula.time_interval.aslist}")
                print(f"INVOLVED_AGENTS: {formula.contributing_agents}")
                print("===================================")
