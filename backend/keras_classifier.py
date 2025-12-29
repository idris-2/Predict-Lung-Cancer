import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin

class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, epochs=50, batch_size=32, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Input
        from keras.optimizers import Adam
        
        X = np.array(X)
        y = np.array(y)
        input_dim = X.shape[1]

        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        X = np.array(X)
        return (self.model.predict(X, verbose=0) > 0.5).astype("int32").flatten()

    def predict_proba(self, X):
        X = np.array(X)
        probs = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - probs, probs])

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                self.model.save(tmp.name)
                tmp_path = tmp.name
            with open(tmp_path, 'rb') as f:
                state['model_weights'] = f.read()
            os.remove(tmp_path)
            state['model'] = None
        return state

    def __setstate__(self, state):
        from keras.models import load_model
        import tempfile
        weights = state.pop('model_weights', None)
        self.__dict__.update(state)
        if weights:
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                tmp.write(weights)
                tmp_path = tmp.name
            self.model = load_model(tmp_path)
            os.remove(tmp_path)
        else:
            self.model = None