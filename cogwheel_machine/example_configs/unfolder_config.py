"""
Example configuration file with parameters for XGBoost.

This file may be copied into ``unfolderdir`` and edited before training
the network.
"""
# kwargs to xgboost.XGBClassifier (except 'num_class' and 'objective')
UNFOLDER_KWARGS = {}

DEVICE = None  # ``None`` will try to use 'cuda' or fall back to 'cpu'.
