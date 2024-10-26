#!/usr/bin/env python3
import casadi as ca
import numpy as np
import os
import sys
from enum import Enum
from itertools import product
from typing import List, Tuple
from itertools import product


class LeadershipToken(Enum) :
    LEADER    = 1
    FOLLOWER  = 2
    UNDEFINED = 0



class Agent:
    """
    Stores information about an agent.

    Attributes:
        id              (int)           : The unique identifier of the agent.
        symbolic_state  (ca.MX)         : The symbolic representation of the agent's state.
        symbolic_input  (ca.MX)         : The symbolic representation of the agent's input.
        state           (np.ndarray)    : The initial state of the agent.
    """
    def __init__(self, id: int, state: np.ndarray, theta: float = None) -> None:
        """
        Initializes an Agent object.

        Args:
            id            (int)         : The unique identifier of the agent.
            initial_state (np.ndarray)  : The initial state of the agent.
        """
        self.id = id
        self.symbolic_state = ca.MX.sym(f'state_{id}', 2)
        self.symbolic_input = ca.MX.sym(f'input_{id}', 2)
        self.state = state
        self.theta = theta if theta is not None else 0.0



class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()



class ImpactSolverLP:
    """
    This is a support class to basically solve a linear program. 
    min Lg^T u or max Lg^T u
    s.t A u <= b
    """
    def __init__(self, agent:Agent, max_velocity:float) -> None:
        
        
        self.Lg    = ca.MX.sym("Lg",2) # This is the parameteric Lie derivative. it will computed at every time outside this function and then it will be given as a parameter
        self.cost  = self.Lg.T @ agent.symbolic_state # change of sign becaise you have to turn maximization into minimization
        
        # (input_)constraints where A,b are from create_approximate_ball_constraints2d
        A, b, input_verticies = create_approximate_ball_constraints2d(radius=max_velocity, points_number=40)
        constraints = A@agent.symbolic_state - b # this is already an ca.MX that is a funciton of the control input
        
        with NoStdStreams():  
            lp          = {'x':agent.symbolic_state, 'f':self.cost, 'g':constraints, 'p':self.Lg} # again we make it parameteric to avoid rebuilding the optimization program since the structure is always the same
            self.solver      = ca.qpsol('S', 'qpoases', lp,{"printLevel":"none"}) # create a solver object 
    
    def maximize(self,Lg:np.ndarray) -> np.ndarray:
        """This program will simply be solved as a function of the parameter. This avoids re bulding the optimization program every time"""
        
        return self.solver(p=-Lg,ubg=0)["x"]
    
    def minimize(self,Lg:np.ndarray) -> np.ndarray:
        """This program will simply be solved as a function of the parameter. This avoids re bulding the optimization program every time"""
        return self.solver(p=Lg,ubg=0)["x"]


def create_rectangular_constraint_function(bounds: List[Tuple[float, float]]) -> Tuple[ca.MX, ca.MX]:
    # Create the base identity matrix for the bounds
    base = np.eye(len(bounds))
    Arows = []
    brows = []
    theta = ca.MX.sym('theta', 1)
    
    # Rotation matrix 
    R = ca.Function('R', [theta], [ca.vertcat(ca.horzcat(ca.cos(theta), -ca.sin(theta)), 
                                              ca.horzcat(ca.sin(theta), ca.cos(theta)))]) 
    for i in range(len(bounds)):
        # Append constraints for the upper and lower bounds

        rotated_base = R(theta) @ base[i, :]
        Arows.append(rotated_base.T)
        Arows.append(-rotated_base.T)

        try:
            brows.append(bounds[i][1])  # Upper bound
            brows.append(-bounds[i][0]) # Lower bound (negated for constraint)
        except:
            raise ValueError(f"Bounds should be a list of lists with two elements. Got {bounds} instead")

        # Print the intermediate results for debugging
        # print(f"Base base: {base}")
        # print(f"Rotated base: {rotated_base}")
        # print("---")


    # Convert lists to numpy arrays for A and b
    A = ca.vertcat(*Arows)
    b = ca.vertcat(*brows)

    A_func = ca.Function('A_func', [theta], [A])
    b_func = ca.Function('b_func', [theta], [b])

    return A_func, b_func




def create_approximate_ball_constraints2d(radius:float,points_number:int)-> Tuple[np.ndarray,np.ndarray,np.ndarray] :
    """
    Computes constraints matrix and vector A,b for an approximation of a ball constraint with a given radius.
    The how many planes will be used to approximate the ball. We cap the value of the planes to 80 to avoid numerical errors in the computation of the normals
    
    Args :
        radius : The radius of the ball
        points_number : The number of planes used to approximate the ball
        
    Returns:
        A : The constraints matrix and vector for the ball constraint
        b : The constraints matrix and vector for the ball constraint
        vertices : The vertices of the polygon that approximates the ball
        
    
    Notes :
    To compute a center shift in the constraints it is sufficient to recompute b as b = b -+ A@center_shift. Indeeed A(x-c)-b = Ax - Ac - b = Ax - (b + Ac) 
    To compute a scaled version of the obtained polygone it is sufficient to scale b  by a scalar gamma> 0. Ax-gamma*b<=0 is the polygon scaled by gamma
    
    """

    # create constraints
    Arows    = []
    brows    = []
    vertices = []
    
    points_number = min(points_number,80) # to avoid numerical error on the computation of the normals
    
    for i in range(0,points_number):
        angle = 2 * np.pi * i / (points_number-1)
        vertices += [np.array([radius*np.cos(angle), radius*np.sin(angle)])]
        
    for j in range(0,len(vertices)) :
        
        tangent = vertices[j] - vertices[j-1]
        norm_tangent = np.linalg.norm(tangent)
        outer_normal = np.array([tangent[1],-tangent[0]])/norm_tangent
        
        b = np.sqrt(radius**2 - (norm_tangent/2)**2)
        Arows += [outer_normal]
        brows += [np.array([[b]])]
    
    A = np.row_stack(Arows)
    b = np.row_stack(brows)
    vertices = np.row_stack(vertices)

    return A,b,vertices


def create_box_constraint_function(bounds: List[Tuple[float,float]])-> Tuple[ca.MX, ca.MX] :
    """
    Create a casadi function that checks if a given vector is inside the box bounds.
    """

    base  = np.eye(len(bounds))
    Arows = []
    brows = []
    # vertices = np.array(list(product(*bounds)))

    for i in range(len(bounds)):
        Arows.append(base[i, :])
        Arows.append(-base[i, :])
        try :
            brows.append(bounds[i][1])
            brows.append(-bounds[i][0])
        except :
            raise ValueError(f"Bounds should be a list of lists with two elements. Got {bounds} instead")

    A = np.row_stack(Arows)
    b = np.row_stack(brows)
    # vertices = np.row_stack(vertices)

    return A,b#,vertices


