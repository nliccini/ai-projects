from DataInterface import getExtraCeditDataSet
from DecisionTree import makeTree, setEntropy, infoGain
import random


def getAverageClassificaionRate(dataset, runs=20, testSize=200, setFunc=setEntropy, infoFunc=infoGain):
    """
    Randomly selects a test set and removes it from the training set.
    """
    scores = []
    examples, attrValues, labelName, labelValues = dataset
    l = len(examples) - 1
    print 'Starting test for average error for %d runs with test size %d' % (runs, testSize)
    for r in xrange(runs):
        runExamples = examples[:]
        test = []
        for i in xrange(testSize):
            test.append(runExamples.pop(random.randint(0, l - i)))
        tree = makeTree(runExamples, attrValues, labelName, setFunc, infoFunc)
        score = evaluateTree(tree, test, labelName)[0]
        print 'Score for run %d is %f' % (r + 1, score)
        scores.append(score)
    average = sum(scores) / float(runs)
    print 'Average classification rate over all runs: %f' % (average)
    return (scores, average)


def evaluateTree(tree, testExamples, labelName):
    """
    Simple function to get the correct classification ratio for a given DTree
    and a set of testing examples.

    Args:
        testExamples (list<dictionary<str,str>>): list of examples to test with
        labelName (str): the name of the label
    Returns:
        tuple<float,
        list<tuple<str,str>>>
        Tuple
    """
    confusion = []
    f = 0.0
    for example in testExamples:
        z = tree.classify(example)
        if example[labelName] == z:
            f += 1.0
        else:
            confusion.append((example[labelName], z))
    return (f / len(testExamples), confusion)


def printDemarcation():
    print 'Done\n____________________________________________________________________\n'


def testHeartDisease(setFunc = setEntropy, infoFunc = infoGain):
    examples,attrValues,labelName,labelValues = getExtraCeditDataSet()
    print 'Testing Heart Disease dataset. Number of examples %d.'%len(examples)
    tree = makeTree(examples, attrValues, labelName, setFunc, infoFunc)
    f = open('heart-disease.out','w')
    f.write(str(tree))
    f.close()
    print 'Tree size: %d.\n'%tree.count()
    print 'Entire tree written out to car.out in local directory\n'
    evaluation = getAverageClassificaionRate((examples,attrValues,labelName,labelValues))
    print 'Results for training set:\n%s\n'%str(evaluation)
    printDemarcation()
    return (tree,evaluation)

def main():
    testHeartDisease()


if __name__=='__main__':
    main()