from math import log
from scipy.stats import chisqprob

class Node:
  """
  A simple node class to build our tree with. It has the following:
  
  children (dictionary<str,Node>): A mapping from attribute value to a child node
  attr (str): The name of the attribute this node classifies by. 
  islead (boolean): whether this is a leaf. False.
  """
  
  def __init__(self,attr):
    self.children = {}
    self.attr = attr
    self.isleaf = False

class LeafNode(Node):
    """
    A basic extension of the Node class with just a value.
    
    value (str): Since this is a leaf node, a final value for the label.
    islead (boolean): whether this is a leaf. True.
    """
    def __init__(self,value):
        self.value = value
        self.isleaf = True
    
class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  Stores the root Node and nothing more. A nice printing method is provided, and
  the function to classify values is left to fill in.
  """
  def __init__(self, root=None):
    self.root = root

  def prettyPrint(self):
    print str(self)
    
  def preorder(self,depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if node.isleaf:
      return '|---'*depth+str(node.value)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.attr),str(val))
      string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
    return string    

  def count(self,node=None):
    if node is None:
      node = self.root
    if node.isleaf:
      return 1
    count = 1
    for child in node.children.values():
      if child is not None:
        count+= self.count(child)
    return count  

  def __str__(self):
    return self.preorder(0, self.root)
  
  def classify(self, classificationData):
    """
    Uses the classification tree with the passed in classificationData.`
    
    Args:
        classificationData (dictionary<string,string>): dictionary of attribute values
    Returns:
        str
        The classification made with this tree.
    """
    node = self.root

    "Traverse through the tree grabbing the node whose value matches the classification data"
    while not node.isleaf:
        if node.attr in classificationData:
            value = classificationData[node.attr]
            node = node.children[value]

    return node.value

  
def getPertinentExamples(examples,attrName,attrValue):
    """
    Helper function to get a subset of a set of examples for a particular assignment 
    of a single attribute. That is, this gets the list of examples that have the value 
    attrValue for the attribute with the name attrName.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValue (str): a value of the attribute
    Returns:
        list<dictionary<str,str>>
        The new list of examples.
    """
    newExamples = []

    for example in examples:
        "Get the value of this attribute for this example"
        value = example[attrName]

        "Only add this example if its value matches the desired value"
        if value == attrValue:
            newExamples = newExamples + [example]

    return newExamples
  
def getClassCounts(examples,className):
    """
    Helper function to get a dictionary of counts of different class values
    in a set of examples. That is, this returns a dictionary where each key 
    in the list corresponds to a possible value of the class and the value
    at that key corresponds to how many times that value of the class 
    occurs.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        className (str): the name of the class
    Returns:
        dictionary<string,int>
        This is a dictionary that for each value of the class has the count
        of that class value in the examples. That is, it maps the class value
        to its count.
    """
    classCounts = {}

    for example in examples:
        "Get the value for this class in this example"
        value = example[className]

        "If it already exists, add another counter to it, else add it to the dict"
        if value in classCounts:
            classCounts[value] += 1
        else:
            classCounts[value] = 1

    return classCounts

def getMostCommonClass(examples,className):
    """
    A freebie function useful later in makeSubtrees. Gets the most common class
    in the examples. See parameters in getClassCounts.
    """
    counts = getClassCounts(examples,className)
    return max(counts, key=counts.get) if len(examples)>0 else None

def getAttributeCounts(examples,attrName,attrValues,className):
    """
    Helper function to get a dictionary of counts of different class values
    corresponding to every possible assignment of the passed in attribute. 
	  That is, this returns a dictionary of dictionaries, where each key  
	  corresponds to a possible value of the attribute named attrName and holds
 	  the counts of different class values for the subset of the examples
 	  that have that assignment of that attribute.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<str>): list of possible values for the attribute
        className (str): the name of the class
    Returns:
        dictionary<str,dictionary<str,int>>
        This is a dictionary that for each value of the attribute has a
        dictionary from class values to class counts, as in getClassCounts
    """
    attributeCounts={}

    for attrValue in attrValues:
        "Get only the examples with this attrValue for this attrName"
        newExamples = getPertinentExamples(examples, attrName, attrValue)

        "Count all the instances of className in these new examples"
        classCounts = getClassCounts(newExamples, className)

        "Add these counts to the dictionary for this attrValue"
        attributeCounts[attrValue] = classCounts

    return attributeCounts


def setEntropy(classCounts):
    """
    Calculates the set entropy value for the given list of class counts.
    This is called H in the book. Note that our labels are not binary,
    so the equations in the book need to be modified accordingly. Note
    that H is written in terms of B, and B is written with the assumption 
    of a binary value. B can easily be modified for a non binary class
    by writing it as a summation over a list of ratios, which is what
    you need to implement.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The set entropy score of this list of class value counts.
    """
    "H = -sum(P(v_k)log_2(P(v_k)))"
    "P(v_k) = classCount / totalCount"
    H = 0.0
    totalCount = float(sum(classCounts))
    for classCount in classCounts:
        q = (classCount / totalCount)
        H += -(q * log(q, 2))
    return H
   

def remainder(examples,attrName,attrValues,className):
    """
    Calculates the remainder value for given attribute and set of examples.
    See the book for the meaning of the remainder in the context of info 
    gain.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The remainder score of this value assignment of the attribute.
    """
    R = 0.0
    attrCounts = getAttributeCounts(examples, attrName, attrValues, className)

    "Compute p + n, the total value of all the attributes in the set"
    totalTrainingSetCount = 0.0
    for attrValue in attrValues:
        classCounts = attrCounts[attrValue].values()
        totalTrainingSetCount += float(sum(classCounts))

    "Compute R = sum((p_k + n_k)/(p + n)B(p_k/(p_k+n_k))"
    for attrValue in attrValues:
        classCounts = attrCounts[attrValue].values()
        totalCount = float(sum(classCounts))
        B = setEntropy(classCounts)
        R += (totalCount / totalTrainingSetCount) * B

    return R
          
def infoGain(examples,attrName,attrValues,className):
    """
    Calculates the info gain value for given attribute and set of examples.
    See the book for the equation - it's a combination of setEntropy and
    remainder (setEntropy replaces B as it is used in the book).

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The gain score of this value assignment of the attribute.
    """
    classCounts = getClassCounts(examples, className).values()
    B = setEntropy(classCounts)

    R = remainder(examples, attrName, attrValues, className)

    gain = B - R
    return gain
  
def giniIndex(classCounts):
    """
    Calculates the gini value for the given list of class counts.
    See equation in instructions.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The gini score of this list of class value counts.
    """
    totalCount = float(sum(classCounts))
    countSum = 0.0
    for classCount in classCounts:
        countSum += (classCount / totalCount) ** 2
    index = 1 - countSum
    return index
  
def giniGain(examples,attrName,attrValues,className):
    """
    Return the inverse of the giniD function described in the instructions.
    The inverse is returned so as to have the highest value correspond 
    to the highest information gain as in entropyGain. If the sum is 0,
    return sys.maxint.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The summed gini index score of this list of class value counts.
    """
    "Get the minimum gini index of the specified attribute"
    giniD = 0.0

    attrCounts = getAttributeCounts(examples, attrName, attrValues, className)
    classCounts = getClassCounts(examples, className)

    "Get the number of all classes"
    "n = sum(S)"
    n = float(sum(classCounts.values()))

    "Compute giniD"
    "S_i = attrCounts[attrValue].values()"
    "n_i = sum(S_i)"
    for attrValue in attrValues:
        classCounts_i = attrCounts[attrValue].values()
        index = giniIndex(classCounts_i)
        n_i = float(sum(classCounts_i))
        giniD += (n_i/n) * index

    "If needed, return the max value to avoid divide by zero errors"
    if giniD == 0.0:
        return float("inf")
    else:
        return 1/giniD

    
def makeTree(examples, attrValues,className,setScoreFunc,gainFunc):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makeSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc))
    
def makeSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attributeValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    "See Artificial Intelligence: A Modern Approach Section 18.3 Figure 18.5 for the pseudocode"

    if len(examples) == 0:
        "if examples is empty then return plurality_value(parent_examples)"
        return LeafNode(defaultLabel)
    elif len(remainingAttributes) == 0:
        "else if attributes is empty then return plurality_value(examples)"
        return LeafNode(getMostCommonClass(examples, className))
    elif len(getClassCounts(examples, className).keys()) == 1:
        "else if all examples have the same classification then return the classification"
        return LeafNode(examples[0][className])
    else:
        "A <-- argmax_{a in attributes}(importance(attrName, examples))"
        max_infoGain = float("-inf")
        best_attrName = None
        for attrName in remainingAttributes:
            infoGain = gainFunc(examples, attrName, attributeValues[attrName], className)
            if infoGain > max_infoGain:
                max_infoGain = infoGain
                best_attrName = attrName
        "tree <-- a new decision tree with root test A"
        tree = Node(best_attrName)
        "for each value v_k of A do"
        best_attrValues = attributeValues[best_attrName]
        new_defaultLabel = getMostCommonClass(examples, className)
        new_remainingAttributes = [attr for attr in remainingAttributes]
        new_remainingAttributes.remove(best_attrName)
        for attrValue in best_attrValues:
            "exs <-- {e : e in examples and e.A = v_k}"
            new_examples = getPertinentExamples(examples, best_attrName, attrValue)
            "subtree <-- decision-tree-learning(exs, attributes - A, examples)"
            subtree = makeSubtrees(new_remainingAttributes, new_examples, attributeValues, className, new_defaultLabel, setScoreFunc, gainFunc)
            "add a branch to tree with label (A = v_k) and subtree (subtree)"
            tree.children[attrValue] = subtree
        return tree


def makePrunedTree(examples, attrValues,className,setScoreFunc,gainFunc,q):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makePrunedSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc,q))
    
def makePrunedSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc,q):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie classEntropy or gini)
        gainFunc (func): the function to score gain of attributes (ie entropyGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    "See Artificial Intelligence: A Modern Approach Section 18.3 Figure 18.5 for the pseudocode"

    if len(examples) == 0:
        "if examples is empty then return plurality_value(parent_examples)"
        return LeafNode(defaultLabel)
    elif len(remainingAttributes) == 0:
        "else if attributes is empty then return plurality_value(examples)"
        return LeafNode(getMostCommonClass(examples, className))
    elif len(getClassCounts(examples, className).keys()) == 1:
        "else if all examples have the same classification then return the classification"
        return LeafNode(examples[0][className])
    else:
        "A <-- argmax_{a in attributes}(importance(attrName, examples))"
        max_infoGain = float("-inf")
        best_attrName = None
        for attrName in remainingAttributes:
            infoGain = gainFunc(examples, attrName, attributeValues[attrName], className)
            if infoGain > max_infoGain:
                max_infoGain = infoGain
                best_attrName = attrName
        best_attrValues = attributeValues[best_attrName]

        "use chi-squared method of determining significance"
        classCounts = getClassCounts(examples, className)
        totalCount = float(sum(classCounts.values()))

        "compute Dev(X)"
        "for each x in X (x = attrVal, X = best_attrName)"
        dev_X = 0.0
        for attrValue in best_attrValues:
            new_examples = getPertinentExamples(examples, best_attrName, attrValue)
            new_classCounts = getClassCounts(new_examples, className)
            D_x = float(sum(new_classCounts.values()))
            dev_p_i = 0.0
            "for each possible value of className"
            for key, classCount in new_classCounts.items():
                p_x = classCount
                "prob of attribute i = count[i] / sum_i(count[i])"
                p_hat = (classCounts[key] / totalCount) * D_x
                dev_p_i += ((p_x - p_hat) ** 2) / p_hat
            "Dev(X) = sum x in X of p_i variance"
            dev_X += dev_p_i

        v = len(best_attrValues)
        prob = chisqprob(dev_X, v-1)
        if prob > q:
            return LeafNode(getMostCommonClass(examples, className))

        "tree <-- a new decision tree with root test A"
        tree = Node(best_attrName)

        "for each value v_k of A do"
        new_defaultLabel = getMostCommonClass(examples, className)
        new_remainingAttributes = [attr for attr in remainingAttributes]
        new_remainingAttributes.remove(best_attrName)
        for attrValue in best_attrValues:
            "exs <-- {e : e in examples and e.A = v_k}"
            new_examples = getPertinentExamples(examples, best_attrName, attrValue)
            "subtree <-- decision-tree-learning(exs, attributes - A, examples)"
            subtree = makePrunedSubtrees(new_remainingAttributes, new_examples, attributeValues, className, new_defaultLabel,
                                   setScoreFunc, gainFunc, q)
            "add a branch to tree with label (A = v_k) and subtree (subtree)"
            tree.children[attrValue] = subtree
        return tree
