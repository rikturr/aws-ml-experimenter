# TO RUN
# python experiments/run_experiment.py experiments/classification_keras.py examples/keras_medicare.py <PEM> <BUCKET> --instance-type p2.xlarge --bid-price 0.40

from keras.models import *
from keras.layers import *

random_state = 42
features_file = 'data/2015_partB_sparse.npz'
labels_file = 'data/2015_partB_lookup.csv'
label_col = 'provider_type'

optimizer = 'sgd'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
batch_size = 32
epochs = 50


def model(input_dim, output_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(500, activation='relu')(input_layer)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu', name='encoded')(x)
    output_layer = Dense(output_dim, activation='softmax')(x)

    return Model(input_layer, output_layer)
