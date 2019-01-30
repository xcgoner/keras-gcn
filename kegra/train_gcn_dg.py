from __future__ import print_function

import numpy as np

# reproducible
from tensorflow import set_random_seed
import random
rnd_seed = 337
np.random.seed(rnd_seed)
set_random_seed(rnd_seed)
random.seed(rnd_seed)

from keras.layers import Input, Dropout, Dot, Subtract, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.optimizer.sgd import AdamAccumulate
from kegra.utils import *
from kegra.augmentation import *
from keras.models import load_model

from keras import backend as K

import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="name of the dataset", default="cora")
parser.add_argument("--nepochs", type=int, help="number of epochs", default=300)
parser.add_argument("--patience", type=int, help="early stopping", default=50)
parser.add_argument("--nfilters", type=int, help="number of hidden features", default=64)
parser.add_argument("--ntrials", type=int, help="number of runs", default=10)
parser.add_argument("--augmentation", type=str, help="type of augmentation: shuffle_edge, shuffle_mix", default="no_augmentation")
parser.add_argument("--shuffle", type=float, help="randomly shuffle edges, percentage", default=0)
parser.add_argument("--alpha", type=float, help="hyperparameter of shuffle_mix", default=0)
parser.add_argument("--nfolds", type=int, help="folds of data augmentation", default=0)
parser.add_argument("--nlayers", type=int, help="number of stacking layers", default=0)
parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
parser.add_argument("--save", type=str, help="path of saved model", default="")

args = parser.parse_args()

print(args, flush=True)

# Define parameters
FILTER = 'localpool'  # 'chebyshev'
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
N_FILTERS = args.nfilters
N_FOLDS = args.nfolds
if args.augmentation == "no_augmentation":
    N_FOLDS = 0

# Get data
X, A, y, edges = load_data(dataset=args.dataset)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

""" Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
print('Using local pooling filters...')
A_ = preprocess_adj(A, SYM_NORM)
# adj for regularization
adj_reg = preprocess_adj(A, SYM_NORM, False)
support = 1

# def add_regularizer(reg_input, output_length):
#     global reg_counter, losses, loss_weights, outputs, reg_mask, sample_weight
#     reg_counter = reg_counter + 1
#     output_name = 'regularization_%02d' % (reg_counter)
#     losses[output_name] = 'mean_squared_error'
#     loss_weights[output_name] = float(args.reigen)
#     outputs[output_name]  = np.zeros((A.shape[0], output_length))
#     sample_weight[output_name] = reg_mask
#     return EigenRegularization(name=output_name)([reg_input, ADJ])

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

    G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
    ADJ = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
    reg_counter = 0
    losses = {'classification': 'categorical_crossentropy'}
    loss_weights = {'classification': 1.0}
    outputs = {'classification': y_train}
    reg_mask = np.array(np.ones_like(train_mask), dtype=np.bool)
    sample_weight = {'classification': train_mask}
    reg_outputs = []

    X_in = Input(shape=(X.shape[1],))

    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(N_FILTERS, support, activation='relu', kernel_regularizer=l2(5e-4))([H, G])


    for i in range(args.nlayers):
        H = Dropout(0.5)(H)
        H = GraphConvolution(N_FILTERS, support, activation='relu', kernel_regularizer=l2(5e-4))([H, G])


    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax', name='classification')([H, G])
    # reg_outputs.append(add_regularizer(Y, y.shape[1]))

    input_list = [X_in, G,]
    output_list = [Y]
    graph = [X, A_]

    # data augmentation
    augmented_graphs = []
    for fold in range(N_FOLDS):
        if args.augmentation == "shuffle_edge":
            shuffled_adj = shuffle_edges(edges, y.shape[0], int(args.shuffle * edges.shape[0]))
        elif args.augmentation == "shuffle_mix":
            shuffled_adj = shuffle_mix(edges, y.shape[0], args.alpha)

        shuffled_conv_adj = preprocess_adj(shuffled_adj, SYM_NORM)
        augmented_graphs.append(shuffled_conv_adj)

    model = Model(inputs=input_list, outputs=output_list)

    # Compile model
    if N_FOLDS > 0:
        # accumulate gradients
        model.compile(loss=losses, 
                loss_weights=loss_weights,
                optimizer=AdamAccumulate(lr=args.lr, accum_iters=N_FOLDS+1))
    else:
        model.compile(loss=losses, 
                    loss_weights=loss_weights,
                    optimizer=Adam(lr=args.lr))

    # reset
    # reset_weights(model)

    # model.summary()

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999
    best_val_acc = 0

    # Fit
    for epoch in range(1, args.nepochs+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, outputs, 
                sample_weight=sample_weight,
                batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        for fold in range(N_FOLDS):
            
            graph_augmented = [X, augmented_graphs[fold]]

            model.fit(graph_augmented, outputs, 
                    sample_weight=sample_weight,
                    batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                    [idx_train, idx_val])
        print("Trial: {:04d}".format(trial+1),
            "Epoch: {:04d}".format(epoch),
            "train_loss= {:.4f}".format(train_val_loss[0]),
            "train_acc= {:.4f}".format(train_val_acc[0]),
            "val_loss= {:.4f}".format(train_val_loss[1]),
            "val_acc= {:.4f}".format(train_val_acc[1]),
            "time= {:.4f}".format(time.time() - t), flush=True)

        if train_val_acc[1] > best_val_acc:
            best_val_acc = train_val_acc[1]
        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
            # save best model
            model.save_weights(args.save)
        else:
            if wait >= args.patience:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1


    # Testing
    model.load_weights(args.save)
    preds = model.predict(graph, batch_size=A.shape[0])
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                [idx_train, idx_val])
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("Test set results:",
        "loss= {:.4f}".format(test_loss[0]),
        "accuracy= {:.4f}".format(test_acc[0]), flush=True)
    test_loss_list.append(test_loss[0])
    test_acc_list.append(test_acc[0])

    print("Avg test set results:",
            "loss= {:.4f} +\- {:.4f}".format(np.mean(test_loss_list), np.std(test_loss_list)),
            "accuracy= {:.4f} +\- {:.4f}".format(np.mean(test_acc_list), np.std(test_acc_list)), flush=True)

