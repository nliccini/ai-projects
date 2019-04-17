# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        "Value Iteration algorithm from Equation (17.6) page 652"
        for iteration in range(0, self.iterations):
            values_prime = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    computed_values = []
                    for action in self.mdp.getPossibleActions(state):
                        value = 0
                        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, next_state)
                            discounted_reward = self.discount * values_prime[next_state]
                            value += probability * (reward + discounted_reward)
                        computed_values = computed_values + [value]
                    self.values[state] = max(computed_values)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, probability in transitions:
            reward = self.mdp.getReward(state, action, next_state)
            discounted_reward = self.discount * self.values[next_state]
            q += probability * (reward + discounted_reward)
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "Check if this is a terminal state"
        if self.mdp.isTerminal(state):
            return None

        "Compute the best action given the precomuted values"
        values = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            value = 0
            transitions = self.mdp.getTransitionStatesAndProbs(state, action)
            for next_state, probability in transitions:
                reward = self.mdp.getReward(state, action, next_state)
                discounted_reward = self.discount * self.values[next_state]
                value += probability * (reward + discounted_reward)
            values[action] = value
        return values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
