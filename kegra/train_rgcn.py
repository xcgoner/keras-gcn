from __future__ import print_function

# reproducible
from numpy.random import seed
from tensorflow import set_random_seed

from keras.layers import Input, Dropout, Dot, Subtract, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution, EigenRegularization
from kegra.utils import *

from keras import backend as K

from keras.legacy import interfaces
from keras.optimizers import Optimizer

class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)  
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import numpy as np

import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="name of the dataset", default="cora")
parser.add_argument("--nepochs", type=int, help="number of epochs", default=300)
parser.add_argument("--patience", type=int, help="early stopping", default=10)
parser.add_argument("--nfilters", type=int, help="number of hidden features", default=64)
parser.add_argument("--ntrials", type=int, help="number of runs", default=10)
parser.add_argument("--reigen", type=float, help="add robust eigen regularizer", default=0)
parser.add_argument("--shuffle", type=float, help="randomly shuffle edges, percentage", default=0)
parser.add_argument("--nfolds", type=int, help="folds of data augmentation", default=1)
parser.add_argument("--nlayers", type=int, help="number of stacking layers", default=0)
parser.add_argument("--lr", type=float, help="learning rate", default=0.01)

args = parser.parse_args()

print(args, flush=True)

# Define parameters
FILTER = 'localpool'  # 'chebyshev'
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
N_FILTERS = args.nfilters
if args.shuffle > 0:
    n_fold_augmentation = args.nfolds
else:
    n_fold_augmentation = 1

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

def add_regularizer(reg_input, output_length):
    global reg_counter, losses, loss_weights, outputs, reg_mask, sample_weight
    reg_counter = reg_counter + 1
    output_name = 'regularization_%02d' % (reg_counter)
    losses[output_name] = 'mean_squared_error'
    loss_weights[output_name] = float(args.reigen)
    outputs[output_name]  = np.zeros((A.shape[0], output_length))
    sample_weight[output_name] = reg_mask
    return EigenRegularization(name=output_name)([reg_input, ADJ])

def reset_weights(model):
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

test_loss_list = []
test_acc_list = []

for trial in range(args.ntrials):

    rnd_seed = 733 + trial
    seed(rnd_seed)
    set_random_seed(rnd_seed)

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

    # regulairzation
    if args.reigen > 0:
        reg_outputs.append(add_regularizer(H, N_FILTERS)) 

    for i in range(args.nlayers):
        H = Dropout(0.5)(H)
        H = GraphConvolution(N_FILTERS, support, activation='relu', kernel_regularizer=l2(5e-4))([H, G])

        # regulairzation
        if args.reigen > 0:
            reg_outputs.append(add_regularizer(H, N_FILTERS)) 


    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax', name='classification')([H, G])
    # reg_outputs.append(add_regularizer(Y, y.shape[1]))

    # define the model
    if args.reigen > 0:
        input_list = [X_in, G, ADJ]
        output_list = [Y] + reg_outputs
        graph = [X, A_, adj_reg]
    else:
        input_list = [X_in, G,]
        output_list = [Y]
        graph = [X, A_]

    model = Model(inputs=input_list, outputs=output_list)

    # Compile model
    if args.shuffle > 0:
        model.compile(loss=losses, 
                loss_weights=loss_weights,
                optimizer=AdamAccumulate(lr=args.lr, accum_iters=n_fold_augmentation))
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

        for fold in range(n_fold_augmentation):
            # data augmentation
            if args.shuffle > 0:
                shuffled_adj = shuffle_edges(edges, y.shape[0], int(args.shuffle * edges.shape[0]))
                shuffled_conv_adj = preprocess_adj(shuffled_adj, SYM_NORM)
                # adj for regularization
                shuffled_reg_adj = preprocess_adj(shuffled_adj, SYM_NORM, False)
                if args.reigen > 0:
                    graph_train = [X, shuffled_conv_adj, shuffled_reg_adj]
                else:
                    graph_train = [X, shuffled_conv_adj]
            else:
                graph_train = graph

            # Single training iteration (we mask nodes without labels for loss calculation)
            model.fit(graph_train, outputs, 
                    sample_weight=sample_weight,
                    batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        if args.reigen > 0:
            preds = preds[0]

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
        else:
            if wait >= args.patience:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1


    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("Test set results:",
        "loss= {:.4f}".format(test_loss[0]),
        "accuracy= {:.4f}".format(test_acc[0]), flush=True)
    test_loss_list.append(test_loss[0])
    test_acc_list.append(test_acc[0])

    print("Avg test set results:",
            "loss= {:.4f} +\- {:.4f}".format(np.mean(test_loss_list), np.std(test_loss_list)),
            "accuracy= {:.4f} +\- {:.4f}".format(np.mean(test_acc_list), np.std(test_acc_list)), flush=True)

