"""Machine learning for QuantLab: PIT datasets, models, purged walk-forward, calibration,
model registry/versioning, meta-model research and model monitoring.

Every ML output is a MODEL_OUTPUT (``core.types.InfoKind``) and a hypothesis until it beats the
simpler baseline out-of-sample and forward. Nothing here places orders or converts an LLM opinion
into a probability.

Module map:
  dataset      build_features / build_labels / build_dataset -> MLDataset (X, y, label_end, fwd_excess)
  models       build_model factory + fit_model -> TrainedModel (estimator + optional calibrator)
  calibration  Calibrator (isotonic | platt), Brier / log-loss / ECE / reliability table
  walkforward  purged + embargoed walk-forward splits, holdout refusal, OOS predictions
  evaluate     AUC / Brier / ECE / precision@top-decile + trading relevance with date-block bootstrap CIs
  registry     ModelRegistry: versioned artifacts, status log, ml_predictions writes
  meta         meta-model research over strategy scores vs best-single and vote-count baselines
  monitor      PSI drift, rolling calibration on matured predictions, status recommendations
"""
