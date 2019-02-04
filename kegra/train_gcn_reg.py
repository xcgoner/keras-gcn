from __future__ import print_function

import numpy as np

# reproducible
from tensorflow import set_random_seed
import random
rnd_seed = 337
np.random.seed(rnd_seed)
set_random_seed(rnd_seed)
random.seed(rnd_seed)

from keras.layers import Input, Dropout, Dot, Subtract, Reshape, Concatenate, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution, DiffRegularization
from kegra.optimizer.sgd import AdamAccumulate
from kegra.utils import *
from kegra.augmentation import *
from keras.models import load_model

from keras import backend as K

import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="name of the dataset", default="cora")
parser.add_argument("--nepochs", type=int, help="number of epochs", default=300)
parser.add_argument("--nfilters", type=int, help="number of hidden features", default=16)
parser.add_argument("--ntrials", type=int, help="number of runs", default=10)
parser.add_argument("--augmentation", type=str, help="type of augmentation: shuffle_edge, shuffle_mix", default="no_augmentation")
parser.add_argument("--shuffle", type=float, help="randomly shuffle edges, percentage", default=0)
parser.add_argument("--alpha", type=float, help="hyperparameter of shuffle_mix", default=0)
parser.add_argument("--nfolds", type=int, help="folds of data augmentation", default=0)
parser.add_argument("--nlayers", type=int, help="number of stacking layers", default=1)
parser.add_argument("--selfloop", type=str, help="type of self-loop", default="eye")
parser.add_argument("--sym", type=int, help="symmetric normalization", default=1)
parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
parser.add_argument("--reigen", type=float, help="weight of regularization", default=0.1)
parser.add_argument("--save", type=str, help="path of saved model", default="")

args = parser.parse_args()

print(args, flush=True)

# Define parameters
if args.sym == 1:
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
else:
    SYM_NORM = False
N_FILTERS = args.nfilters
N_FOLDS = args.nfolds
if args.augmentation == "no_augmentation":
    N_FOLDS = 0

# Get data
nodes, edges, A, X, y_train, y_val, y_test, idx_train, idx_val, idx_test = load_data(args.dataset)

# Parameters
N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
n_classes = y_train.shape[1]  # Number of classes

# Preprocessing operations
X = preprocess_features(X)
A_ = preprocess_adj(A, SYM_NORM, args.selfloop)

# # Normalize X
# X /= X.sum(1).reshape(-1, 1)

""" Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
print('Using local pooling filters...')

support = 1

def add_regularizer(x, Px, output_length):
    global reg_counter, losses, loss_weights, train_outputs, val_outputs, test_outputs, reg_mask, train_sample_weight, val_sample_weight, test_sample_weight, N, weighted_metrics
    reg_counter = reg_counter + 1
    output_name = 'regularization_%02d' % (reg_counter)
    losses[output_name] = 'mean_squared_error'
    loss_weights[output_name] = float(args.reigen)
    train_outputs[output_name]  = np.zeros((N, output_length))
    val_outputs[output_name]  = np.zeros((N, output_length))
    test_outputs[output_name]  = np.zeros((N, output_length))
    train_sample_weight[output_name] = reg_mask
    val_sample_weight[output_name] = reg_mask
    test_sample_weight[output_name] = reg_mask
    weighted_metrics.append('mse')
    return DiffRegularization(name=output_name)([x, Px])

def reset_weights(model):
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

test_loss_list = []
test_acc_list = []

for trial in range(args.ntrials):

    rnd_seed = 733 + trial
    np.random.seed(rnd_seed)
    set_random_seed(rnd_seed)
    random.seed(rnd_seed)

    # A_ will be passed to G, which is the normalized adjacency matrix with self-loop
    G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)

    # feature input
    X_in = Input(shape=(F,))

    # multiple outputs
    reg_counter = 0
    losses = {'classification': 'categorical_crossentropy'}
    loss_weights = {'classification': 1.0}
    train_outputs = {'classification': y_train}
    val_outputs = {'classification': y_val}
    test_outputs = {'classification': y_test}
    reg_mask = np.array(np.ones_like(idx_train), dtype=np.bool)
    train_sample_weight = {'classification': idx_train}
    val_sample_weight = {'classification': idx_val}
    test_sample_weight = {'classification': idx_test}
    weighted_metrics=['acc']
    reg_outputs = []

    # Define model architecture
    # The model is similar to https://github.com/dmlc/dgl/blob/master/examples/mxnet/gcn/gcn_concat.py
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.

    # # Dense input
    # H = Dense(N_FILTERS, activation='relu', kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))(X_in)
    # # H = Dense(N_FILTERS, activation='relu')(X_in)
    # H = Dropout(0.5)(H)
    # H = Concatenate()([X_in, H])

    activation = 'tanh'

    H_in = Dense(N_FILTERS, activation=activation, kernel_regularizer=l2(5e-4))(X_in)
    H = H_in

    for i in range(args.nlayers):
        H_input = H
        H = GraphConvolution(N_FILTERS, support, activation=activation, kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4))([H_input, G])
        # H = GraphConvolution(N_FILTERS, support, activation='relu')([H_input, G])
        H = Dropout(0.5)(H)
        # H = Concatenate()([H_input, H])

    Y = Concatenate()([H, H_in])
    Y = Dense(n_classes, activation='softmax', kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4), name='classification')(H)
    # Y = Dense(n_classes, activation='softmax')(H)
    # Y = GraphConvolution(n_classes, support, activation='softmax', kernel_regularizer=l2(5e-4))([H, G])
    if args.reigen > 0:
        reg_outputs.append(add_regularizer(H_in, H, N_FILTERS))

    input_list = [X_in, G,]
    output_list = [Y]
    graph = [X, A_]

    if args.reigen > 0:
        output_list = output_list + reg_outputs

    # data augmentation
    augmented_graphs = []
    for fold in range(N_FOLDS):
        if args.augmentation == "shuffle_edge":
            shuffled_adj = shuffle_edges(nodes, edges, N, int(args.shuffle * edges.shape[0]))
        elif args.augmentation == "shuffle_mix":
            shuffled_adj = shuffle_mix(nodes, edges, N, args.alpha)

        shuffled_conv_adj = preprocess_adj(shuffled_adj, SYM_NORM, args.selfloop)
        augmented_graphs.append(shuffled_conv_adj)

    model = Model(inputs=input_list, outputs=output_list)

    # Compile model
    if N_FOLDS > 0:
        # accumulate gradients
        optimizer = AdamAccumulate(lr=args.lr, accum_iters=N_FOLDS+1)
    else:
        optimizer = Adam(lr=args.lr)
    model.compile(loss=losses, 
                  loss_weights=loss_weights,
                  weighted_metrics=weighted_metrics,
                  optimizer=optimizer)

    # reset
    # reset_weights(model)

    # model.summary()

    # Helper variables for main training loop
    preds = None
    best_val_loss = 99999

    # Fit
    for epoch in range(1, args.nepochs+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, train_outputs, 
                sample_weight=train_sample_weight,
                batch_size=N, epochs=1, shuffle=False, verbose=0)

        for fold in range(N_FOLDS):
            
            graph_augmented = [X, augmented_graphs[fold]]

            model.fit(graph_augmented, train_outputs, 
                    sample_weight=train_sample_weight,
                    batch_size=N, epochs=1, shuffle=False, verbose=0)

        # Evaluate model
        # print(model.metrics_names, flush=True)
        eval_results = model.evaluate(graph, val_outputs,
                                    sample_weight=val_sample_weight,
                                    batch_size=N, verbose=0)
        if args.reigen > 0:
            val_loss = eval_results[1]
            val_acc = eval_results[3]
        else:
            val_loss = eval_results[0]
            val_acc = eval_results[1]
        print("Trial: {:04d}".format(trial+1),
            "Epoch: {:04d}".format(epoch),
            "val_loss= {:.4f}".format(val_loss),
            "val_acc= {:.4f}".format(val_acc),
            "time= {:.4f}".format(time.time() - t), flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save best model
            model.save_weights(args.save)


    # Testing
    model.load_weights(args.save)
    # Evaluate model
    eval_results = model.evaluate(graph, test_outputs,
                                sample_weight=test_sample_weight,
                                batch_size=N, verbose=0)
    if args.reigen > 0:
        test_loss = eval_results[1]
        test_acc = eval_results[3]
    else:
        test_loss = eval_results[0]
        test_acc = eval_results[1]
    print("Test set results:",
        "loss= {:.4f}".format(test_loss),
        "accuracy= {:.4f}".format(test_acc), flush=True)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

    print("Avg test set results:",
            "loss= {:.4f} +\- {:.4f}".format(np.mean(test_loss_list), np.std(test_loss_list)),
            "accuracy= {:.4f} +\- {:.4f}".format(np.mean(test_acc_list), np.std(test_acc_list)), flush=True)

