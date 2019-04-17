# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.1  # More focused on immediate reward
    answerNoise = 0  # Allows agent to go straight there (through cliff path)
    answerLivingReward = -1  # Want to reach any terminal state ASAP
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.25  # Focused on immediate reward more than long term
    answerNoise = 0.2  # Allows agent to accidentally take longer path
    answerLivingReward = -1  # Want to reach any terminal state ASAP
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 0.5  # More focused on long term reward than immediate
    answerNoise = 0  # Allows agent to go straight there (through cliff path)
    answerLivingReward = -0.5  # Want to reach a terminal state soon
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.9  # Much more focused on long term reward
    answerNoise = 0.2  # Allows agent to accidentally take longer path
    answerLivingReward = -0.5  # Want to reach a terminal state soon
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 0.9  # Focus on long term reward: being alive
    answerNoise = 0.2  # Agent can wander, it doesn't matter
    answerLivingReward = 10  # It's too good being alive to terminate
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    # There simply aren't enough learning episodes to solve this problem
    # (50 episodes is a relatively small number in terms of reinforcement learning)
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
