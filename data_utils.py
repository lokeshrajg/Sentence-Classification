import numpy as np


class Data(object):

    def __init__(self,
                 data_source,
                 label_source,
                 alphabet="abcdefghijklmnopqrstuvwxyz",
                 l0=512,
                 batch_size=128,
                 no_of_classes=12):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1

        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source
        self.label_source = label_source

    def loadData(self):
        data = []
        feature_values = []
        label = []
        with open(self.data_source) as xtrain_f, open(self.label_source) as ytrain_f:
            for line in xtrain_f:
                line = line.rstrip("\n")
                feature_values.append(line)
            for line in ytrain_f:
                line = line.rstrip("\n")
                label.append(line)
        for i in range(len(label)):
            data.append((int(label[i]), feature_values[i]))
        self.data = np.array(data)
        self.shuffled_data = self.data

    def loadTestData(self):
        test_data = []
        with open(self.data_source) as xtest_f:
            for line in xtest_f:
                line = line.rstrip("\n")
                test_data.append(line)
        self.test_data = np.array(test_data)

    def shuffleData(self):
        np.random.seed(235)
        data_size = len(self.data)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]

    def getBatch(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = min((batch_num + 1) * self.batch_size, data_size)
        return self.shuffled_data[start_index:end_index]

    def getBatchToIndices(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def getAllData(self):
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def getTestAllData(self):
        data_size = len(self.test_data)
        start_index = 0
        end_index = data_size
        batch_texts = self.test_data[start_index:end_index]
        batch_indices = []
        for s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
        return np.asarray(batch_indices, dtype='int64')

    def strToIndexs(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        k = 0
        for i in range(1, n + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]

        return str2idx

    def getLength(self):
        return len(self.data)


if __name__ == '__main__':
    data = Data()
    data.loadData()
    with open("test.vec", "w") as fo:
        for i in range(data.getLength()):
            c = data.data[i][0]
            txt = data.data[i][1]
            vec = ",".join(map(str, data.strToIndexs(txt)))
            fo.write("{}\t{}\n".format(c, vec))