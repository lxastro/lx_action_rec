import struct
import numpy as np
from common import floatX

class DataEngine(object):
    def __init__(self):
        self.length = 25
        self.feature_dim = 4096
        self.nb_class = 101
    
    def load(self, feature_path, label_path):
        fileData = open(feature_path, 'rb')
        rows = struct.unpack('i', fileData.read(4))[0]
        cols = struct.unpack('i', fileData.read(4))[0]
        _ = struct.unpack('i', fileData.read(4))[0]
        self.nb_samples = rows
        
        self.feature = np.zeros((self.nb_samples, self.length, self.feature_dim), floatX)
        for i in range(self.nb_samples):
            self.feature[i, :, :] = (np.array(struct.unpack('f' * cols, fileData.read(4 * cols)), dtype = floatX)).reshape((self.length, self.feature_dim))
        fileData.close()


        fileData = open(label_path, 'r')
        temp_label = fileData.readlines()
        self.int_label = np.zeros(len(temp_label), np.int)
        for i in range(len(temp_label)):
            self.int_label[i] = int(temp_label[i])
        fileData.close()
        self.label = np.zeros((self.nb_samples, self.nb_class))
        for i in range(self.nb_samples):
            self.label[i, self.int_label[i]] = 1
        
    def get(self, ids):
        return self.feature[ids,:,:], self.label[ids]
    
    def count_label(self):
        self.cnt_label = np.zeros((self.nb_class,), 'int')
        for l in self.int_label:
            self.cnt_label[l] = self.cnt_label[l] + 1
        return self.cnt_label
    
    def get_label_str(self, label_vec, min_prob = 1-1e-5):
        s = ''
        for i,p in enumerate(label_vec):
            if p>min_prob:
                s += '%3d:%5.3f'%(i,p)
        return s

if __name__ == '__main__':
    import common
    import sys
    test_data = DataEngine()
    test_data.load(common.test_feature_path, common.test_label_path)
    print test_data.feature.shape
    print test_data.label.shape
    
    for i in range(1):
        print test_data.get(i)[0]
        print test_data.get(i)[0].shape
        print test_data.get(i)[1]
        print '-'*50
        
    cnt = test_data.count_label()
    for i in range(test_data.nb_class):
        if (1+i)%8 == 0:
            print
        sys.stdout.write("%3d:%2d  " %(i, cnt[i]))
    
    print test_data.label
    
    s = test_data.get_label_str(test_data.label[95])
    print s
    
    