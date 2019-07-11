import pandas as pd

class DecisionTreeClassifier:

    def run(self, train, test, max_depth, min_size):
        tree = self.buildTree(train, max_depth, min_size)
        predictions = list()
        for row in test:
            prediction = self.predict(tree, row)
            predictions.append(prediction)
        return predictions

    def giniIndex(self, groups, classes):
        intancesNumber = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for classVal in classes:
                p = [row[-1] for row in group].count(classVal) / size
                score += p * p
            gini += (1.0 - score) * (size / intancesNumber)
        return gini

    def testSplit(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def getSplit(self, dataset):
        classValues = list(set(row[-1] for row in dataset))
        bIndex, bValue, bScore, bGroups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.testSplit(index, row[index], dataset)
                gini = self.giniIndex(groups, classValues)
                if (gini < bScore):
                    bIndex, bValue, bScore, bGroups = index, row[index], gini, groups
        return {'index': bIndex, 'value': bValue, 'groups': bGroups}

    def toTerminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, maxDepth, minSize, depth):
        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.toTerminal(left + right)
            return
        if (depth >= maxDepth):
            node['left'], node['right'] = self.toTerminal(left), self.toTerminal(right)
            return
        if (len(left) <= minSize):
            node['left'] = self.toTerminal(left)
        else:
            node['left'] = self.getSplit(left)
            self.split(node['left'], maxDepth, minSize, depth + 1)
        if len(right) <= minSize:
            node['right'] = self.toTerminal(right)
        else:
            node['right'] = self.getSplit(right)
            self.split(node['right'], maxDepth, minSize, depth + 1)

    def buildTree(self, train, maxDepth, minSize):
        root = self.getSplit(train)
        self.split(root, maxDepth, minSize, 1)
        return root

    def predict(self, node, row):
        if (row[node['index']] < node['value']):
            if (isinstance(node['left'], dict)):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if (isinstance(node['right'], dict)):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def printTree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * '   ', (node['index'] + 1), node['value'])))
            self.printTree(node['left'], depth + 1)
            self.printTree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * '   ', node)))

def removeIrrelevantData(vector, irrelevantData):
    vector.drop(irrelevantData, axis=1, inplace=True)
    return vector

def dummyData(vector):
    return pd.get_dummies(vector)

def verifyNullValues(vector):
    print(vector.isnull().sum().sort_values(ascending=False).head(10))

def updateUsingMean(vector, colunmString):
    vector[colunmString].fillna(vector[colunmString].mean(), inplace=True)

def getDataWithoutExpectedResult(vector, columnName):
    return vector.drop(columnName, axis=1)

def getExpectedResult(vector, columnName):
    return vector[columnName]

print('For use the algorithm is necessary the archives: ')
print('Path: archives/train.csv -> For build the Tree')
print('Path: archives/test.csv -> For test the Tree')
print('Path: archives/gender_submission.csv -> For verify accuracy of the result')
print('')
input("Press Enter to continue...")
train = pd.read_csv('archives/train.csv')
test = pd.read_csv('archives/test.csv')
verificationTest = pd.read_csv('archives/gender_submission.csv')

print('Removing irrelevant data ...')
train = removeIrrelevantData(train, ['Name', 'Ticket', 'Cabin', 'PassengerId'])
test = removeIrrelevantData(test, ['Name', 'Ticket', 'Cabin'])
verificationTest = removeIrrelevantData(verificationTest, ['PassengerId'])

print('Applying the one-shot-encoding ...')
trainDummy = dummyData(train)
testDummy = dummyData(test)

print('Updating null data with mean values ...')
updateUsingMean(trainDummy, 'Age')
updateUsingMean(testDummy, 'Age')
updateUsingMean(testDummy, 'Fare')

print('Sorting the data... ')
trainOrdened = trainDummy[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S", "Survived"]]
testOrdened = testDummy[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]]
testValues = verificationTest.values

print('----------------------------------------------------------------------')
print('')
maxDepth = int(input('Insert the Max Tree length: '))
print('')

print('Building the tree...')
decisionTree = DecisionTreeClassifier()
prediction = decisionTree.run(trainOrdened.values, testOrdened.values, maxDepth, 1)
print('Build successfully !')

print('')
input('Press enter to continue and see Tree Generated...')
print('')

print('---------------------------- Tree Generated --------------------------')
root = decisionTree.buildTree(trainOrdened.values, 3, 1)
decisionTree.printTree(root, 3)
print('----------------------------------------------------------------------')
print('LEGEND:')
print(' X -> Feature')
print(' < -> Value of the Gini Index')
count = 1
for i in testOrdened.columns:
    print(' ', i, ' -> ', count)
    count += 1

input('Press enter to continue and see Accuracy...')
print('')
print('Accuracy: ', decisionTree.accuracy(prediction,testValues))

print('')
input('Press enter to continue and see Expected Result and Got...')
for i in range(len(prediction)):
    print('Expected=%d Got=%d' % (testValues[i][0], prediction[i]))

intPrediction = []
for i in range(len(prediction)):
    intPrediction.append(int(prediction[i]))

print('')
print('Generating a submission csv file ...')

submission = pd.DataFrame()
submission['PassengerId'] = testDummy['PassengerId']
submission['Survived'] = intPrediction

submission.to_csv('result/submission.csv', index=False)

print('Submission csv file generated !')