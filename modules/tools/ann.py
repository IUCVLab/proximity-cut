# by Stanislav Protasov
# based on
# https://stackoverflow.com/questions/46728937/retrieve-final-hidden-activation-layer-output-from-sklearns-mlpclassifier

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import ACTIVATIONS

def forward(clf, X):
    ''' @clf - pretrained MLPClassifier from sklearn '''
    ''' @X - input vector '''
    ''' @returns - list of layer activations '''
    hidden_layer_sizes = list(clf.hidden_layer_sizes)
    hidden_activation = ACTIVATIONS[clf.activation]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(safe_sparse_dot(activations[i], clf.coefs_[i]))
        activations[i+1] += clf.intercepts_[i]
        if (i + 1) != (clf.n_layers_ - 1):
            v = hidden_activation(activations[i+1])

    return activations