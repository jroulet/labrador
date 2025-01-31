import numpy as np
import xgboost as xgb


class UnfoldingClassifier:
    def __init__(self, rescaled_parameters, compressed_data, **xgb_params):
        self.rescaled_parameters = rescaled_parameters
        self.compressed_data = compressed_data
        self.input_data = np.hstack((self.rescaled_parameters, self.compressed_data))
        self.booster = None
        self.xgb_params = xgb_params

    def train(self, num_boost_round=5000):
        """
        Train the XGBoost model.
        """
        if self.xgb_params is None:
            self.xgb_params = {
                'max_depth': 6,  # the maximum depth of each tree
                'eta': 0.1,  # the training step for each iteration
                'silent': 1,  # logging mode - quiet
                'objective': 'multi:softprob',  # error evaluation for multiclass training
                'num_class': 16  # the number of classification classes
            }
        self.booster = xgb.train(
            self.xgb_params, 
            xgb.DMatrix(self.input_data), 
            num_boost_round=num_boost_round
        )
        
    def predict(self, test_data):
        """
        Make predictions from the trained XGBoost model.
        """
        if self.booster is None:
            raise RuntimeError("Model not trained yet.")
        predictions = self.booster.predict(xgb.DMatrix(test_data))
        
        return predictions.reshape(test_data.shape[0], 16)
    
    def confusion_matrix(self, predictions, true_labels, **filename):
        """
        Compute the confusion matrix for the given predictions.
        """
        confusion_matrix = np.zeros((16, 16))
        
        # Populate the probabilistic confusion matrix
        for true_label, probabilities in zip(true_labels, predictions):
            confusion_matrix[true_label] += probabilities
        
        # Normalize rows for better interpretation (optional)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_confusion_matrix = confusion_matrix / (row_sums + 1e-9)  # Avoid division by zero
        
        return normalized_confusion_matrix
    
    def save(self, filename):
        """
        Save a class instance.
        """
        np.savez_compressed(
            filename, 
            rescaled_parameters=self.rescaled_parameters, 
            compressed_data=self.compressed_data, 
            xgb_params=self.xgb_params
        )
    
    @classmethod
    def load(cls, filename):
        """
        Load a class instance.
        """
        data = np.load(filename, allow_pickle=True)
        instance = cls(
                rescaled_parameters=data['rescaled_parameters'], 
                compressed_data=data['compressed_data'], 
                **data['xgb_params'].item()
            )
        return instance
        
    




