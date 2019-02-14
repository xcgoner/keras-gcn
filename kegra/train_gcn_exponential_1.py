from __future__ import print_function

import numpy as np

# reproducible
from tensorflow import set_random_seed
import random
rnd_seed = 337
np.random.seed(rnd_seed)
set_random_seed(rnd_seed)
random.seed(rnd_seed)

from keras.layers import Input, Dropout, Dot, Subtract, Reshape, Concatenate, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.optimizer.sgd import AdamAccumulate
from kegra.utils import *
from keras.models import load_model

from keras import backend as K

import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="name of the dataset", default="cora")
parser.add_argument("--train-percent", type=float, help="percentage of training data", default=0.036)
parser.add_argument("--nepochs", type=int, help="number of epochs", default=300)
parser.add_argument("--nfilters", type=int, help="number of hidden features", default=16)
parser.add_argument("--ntrials", type=int, help="number of runs", default=10)
parser.add_argument("--nlayers", type=int, help="number of stacking layers", default=1)
parser.add_argument("--expm", type=int, help="order of matrix exponential", default=2)
parser.add_argument("--sym", type=int, help="symmetric normalization", default=1)
parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
parser.add_argument("--append", type=int, help="append to test set", default=0)
parser.add_argument("--save", type=str, help="path of saved model", default="")

args = parser.parse_args()

print(args, flush=True)

# Define parameters
if args.sym == 1:
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
else:
    SYM_NORM = False
N_FILTERS = args.nfilters

# Get data
nodes, edges, A, X, y_train_origin, y_val_origin, y_test_origin, mask_train, mask_val, mask_test = load_data(args.dataset)

# # idx_val += idx_train

# idx_test += idx_val
# idx_val = idx_train
# y_test += y_val
# y_val = y_train


# Parameters
N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
n_classes = y_train_origin.shape[1]  # Number of classes

# Preprocessing operations
X = preprocess_features(X)

# Matrix exponential
A_exp = approx_expm(A, args.expm)
A_exp = preprocess_adj(A_exp, SYM_NORM, 'none')


# Data statistics
print("Classes {:03d}, Train samples {:.2%}, Val samples {:.2%}, Test samples {:.2%}" .format(n_classes, np.sum(mask_train)/mask_train.shape[0], np.sum(mask_val)/mask_val.shape[0], np.sum(mask_test)/mask_test.shape[0]), flush=True)
# print("Classes {:03d}, Train samples {:.2%}, Val samples {:.2%}, Test samples {:.2%}" .format(n_classes, np.sum(mask_train), np.sum(mask_val), np.sum(mask_test)), flush=True)

# mask_test += mask_val
# y_test_origin += y_val_origin

# # Normalize X
# X /= X.sum(1).reshape(-1, 1)

""" Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
print('Using local pooling filters...')

support = 1

def reset_weights(model):
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

test_loss_list = []
test_acc_list = []

for trial in range(args.ntrials):

    idx_train, idx_test, y_train, y_test = split_train_test(mask_train, mask_test, y_train_origin, y_test_origin, args.train_percent, args.append)
    # # for experiments
    # idx_val = idx_test
    # y_val = y_test
    idx_val = mask_val
    y_val = y_val_origin

    # A_ will be passed to G, which is the normalized adjacency matrix with self-loop
    G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
    # ADJ = Input(shape=(None, None), batch_shape=(None, None), sparse=True)

    # feature input
    X_in = Input(shape=(F,))

    # Define model architecture
    # The model is similar to https://github.com/dmlc/dgl/blob/master/examples/mxnet/gcn/gcn_concat.py
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.

    H = X_in

    for i in range(args.nlayers-1):
        H = GraphConvolution(N_FILTERS, support, activation='relu', kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))([H, G])
        H = Dropout(0.5)(H)

    Y = GraphConvolution(n_classes, support, activation='softmax', kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))([H, G])

    input_list = [X_in, G,]
    output_list = Y
    graph = [X, A_exp]

    model = Model(inputs=input_list, outputs=output_list)

    # Compile model
    optimizer = Adam(lr=args.lr)
    model.compile(loss='categorical_crossentropy', 
                  weighted_metrics=['acc'],
                  optimizer=optimizer)

    # reset
    # reset_weights(model)

    # model.summary()

    # Helper variables for main training loop
    preds = None
    best_val_loss = 99999
    best_val_acc = 0

    # Fit
    for epoch in range(1, args.nepochs+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, 
                sample_weight=idx_train,
                batch_size=N, epochs=1, shuffle=False, verbose=0)

        # Evaluate model
        eval_results = model.evaluate(graph, y_val,
                                    sample_weight=idx_val,
                                    batch_size=N, verbose=0)
        print("Trial: {:04d}".format(trial+1),
            "Epoch: {:04d}".format(epoch),
            "val_loss= {:.4f}".format(eval_results[0]),
            "val_acc= {:.4f}".format(eval_results[1]),
            "time= {:.4f}".format(time.time() - t), flush=True)

        # Save best
        # if eval_results[0] < best_val_loss:
        #     best_val_loss = eval_results[0]
        #     # save best model
        #     model.save_weights(args.save)
        if eval_results[1] > best_val_acc:
            best_val_acc = eval_results[1]
            # save best model
            model.save_weights(args.save)


    # Testing
    model.load_weights(args.save)
    # Evaluate model
    eval_results = model.evaluate(graph, y_test,
                                sample_weight=idx_test,
                                batch_size=N, verbose=0)
    print("Test set results:",
        "loss= {:.4f}".format(eval_results[0]),
        "accuracy= {:.4f}".format(eval_results[1]), flush=True)
    test_loss_list.append(eval_results[0])
    test_acc_list.append(eval_results[1])

    print("Avg test set results:",
            "loss= {:.4f} +\- {:.4f}".format(np.mean(test_loss_list), np.std(test_loss_list)),
            "accuracy= {:.4f} +\- {:.4f}".format(np.mean(test_acc_list), np.std(test_acc_list)), flush=True)

