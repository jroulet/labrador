"""
Example configuration file with parameters for XGBoost.

This file may be copied into ``unfolderdir`` and edited before training
the network.
"""
UNFOLDER_KWARGS = {
    'num_class': 2 ** len(TRANSFORM_CLASS.folded_params),
    'objective': 'multi:softprob',
}  # kwargs to xgboost.XGBClassifier

DEVICE = None  # ``None`` will try to use 'cuda' or fall back to 'cpu'.
