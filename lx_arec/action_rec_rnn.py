from __future__ import print_function
import dlx.unit.core as U
import dlx.unit.recurrent as R
from dlx.model import Model
from data import DataEngine
import common
import numpy as np
np.random.seed(1337)

'''Load Data'''
print ('load training data...') 
train = DataEngine()
train.load(common.train_feature_path, common.train_label_path)
print ('load testing data...')
test = DataEngine()
test.load(common.test_feature_path, common.test_label_path)

feature_dim = train.feature_dim
length = train.length
n_class = train.nb_class
n_hidden = 1024
batch_size = 40

'''Define Units'''
#Data unit
data = U.Input(3, 'X')
#RNN unit
rnn = R.LSTM(length, feature_dim, n_hidden, name='ENCODER')
#rnn = R.RNN(length, feature_dim, n_hidden, name='ENCODER')
#Time Distributed Dense
tdd = U.TimeDistributedDense(length, n_hidden, n_class, 'TDD')
tdm = U.TimeDistributedMerge('ave')
#Activation
activation = U.Activation('softmax')
#Output layer
output = U.Output()

'''Define Relations'''
rnn.set_input('input_sequence', data, 'output')
tdd.set_input('input', rnn, 'output_sequence')
tdm.set_input('input', tdd, 'output')
activation.set_input('input', tdm, 'output')
output.set_input('input', activation, 'output')

'''Build Model'''
model = Model()
model.add_input(data, 'X')
model.add_output(output, 'y')
model.add_hidden(rnn)
model.add_hidden(tdd)
model.add_hidden(tdm)
model.add_hidden(activation)


model.compile(optimizer='rmsprop', loss_configures = [('y', 'categorical_crossentropy', None, False, "categorical"),], verbose=0)

score = model.evaluate(data = {'X': test.feature, 
                           'y': test.label},
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Train the model
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(data = {'X': train.feature, 
                      'y': train.label},
              batch_size=batch_size, 
              nb_epoch=1,
              show_accuracy=True, 
              shuffle=True,
              verbose=-5*batch_size,
              #validation_split=0.1
              validation_data = {'X': test.feature, 'y': test.label}
              )
        ###
    # Select 10 samples from the validation set at random so we can visualize errors
    print('---------------------------')
    for i in range(10):
        ind = np.random.randint(0, len(test.feature))
        rowX, rowy = test.feature[np.array([ind])], test.label[np.array([ind])]
        preds = model.predict({'X': rowX}, class_mode = {'y': None}, verbose=0)['y']
        correct = test.get_label_str(rowy[0])
        guess = test.get_label_str(preds[0], min_prob=0.05)
    
        print('Correct:', correct)
        print('Predict:', guess)
        print('---------------------------')
        
        
        
        