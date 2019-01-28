from __future__ import print_function

from keras.layers import Input, Dropout, Dot, Subtract, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution, EigenRegularization
from kegra.utils import *

from keras import backend as K

import numpy as np

import time

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 500
PATIENCE = 50  # early stopping patience
N_FILTERS = 16

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

""" Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
print('Using local pooling filters...')
A_ = preprocess_adj(A, SYM_NORM)
# adj for regularization
adj_reg = preprocess_adj(A, SYM_NORM, False)
adj_reg = adj_reg.toarray()
support = 1
graph = [X, A_, adj_reg]
G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
ADJ = Input(shape=(None, None), batch_shape=(None, None))



X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(N_FILTERS, support, activation='relu', kernel_regularizer=l2(5e-4))([H, G])

# regulairzation
# reg_output = Dot(axes=0)([H, ADJ])
reg_output = EigenRegularization(name='regularization')([H, ADJ])


H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax', name='classification')([H, G])

# Compile model
model = Model(inputs=[X_in, G, ADJ], outputs=[Y, reg_output])
model.compile(loss={'classification': 'categorical_crossentropy', 
                    'regularization': 'mean_squared_error'}, 
              loss_weights={'classification': 1.0,
                            'regularization': 0.01},
              metrics={'classification': 'accuracy',
                        'regularization': 'mean_squared_error'},
              optimizer=Adam(lr=0.01))

model.summary()

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, {'classification': y_train, 'regularization': np.zeros((A.shape[0], N_FILTERS))}, 
              sample_weight={'classification': train_mask, 'regularization': np.array(np.ones_like(train_mask), dtype=np.bool)},
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds, _ = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t), flush=True)

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
