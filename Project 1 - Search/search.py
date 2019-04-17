# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def genericSearch(problem, frontier, heuristic=None):
    """
    Generic Search Algorithm that adjusts for informed vs uninformed searching
    strategies

    Here a Search Node is taken to be (state, actions)

    :param problem: the SearchProblem context
    :param frontier: the data structure used for storing new nodes
    :param heuristic: the optional heuristic for informed searching
    :return: a list of actions to perform
    """
    explored = []  # The list of explored states to make searching complete
    solution = []  # The list of actions representing the solved search

    "Initialize the correct data structure to use for the search"
    "Informed searching uses some sort of priority system, here a PriorityQueue"
    informedSearch = isinstance(frontier, util.PriorityQueue)
    if informedSearch:
        frontier.push((problem.getStartState(), solution), heuristic(problem.getStartState(), problem))
    else:
        frontier.push((problem.getStartState(), solution))

    while not frontier.isEmpty():
        "Pop the node from the frontier"
        state, solution = frontier.pop()

        "Check if this state is the goal state, if so return the solution"
        if problem.isGoalState(state):
            return solution

        if state not in explored:
            "Consider this node explored"
            explored.append(state)

            "Check all its successor nodes"
            successors = problem.getSuccessors(state)
            for successor in successors:
                state, action, cost = successor

                "Append the action that got to this node to the current solution"
                actions = solution + [action]

                "Push the node using the appropriate search style (informed vs uninformed)"
                if informedSearch:
                    f = problem.getCostOfActions(actions) + heuristic(state, problem)
                    frontier.push((state, actions), f)
                else:
                    frontier.push((state, actions))

    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return genericSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    return genericSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    "UCS is just A* search but with a trivial heuristic"
    return aStarSearch(problem)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    return genericSearch(problem, util.PriorityQueue(), heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
