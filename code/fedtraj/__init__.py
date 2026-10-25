"""fedtraj: federated vehicle-trajectory prediction: Fed-SSBR (curriculum-quota scheduling and update-deviation
reweighting), and FedTR (scale-consistent aggregation and tail routing).

Modules
    layout      44-channel frame layout, causal kinematics, windows, splits, partitions, dataset writer
    prep_ngsim  NGSIM US-101 / I-80 raw trajectories -> dataset directory
    prep_highd  highD recordings -> dataset directory
    model       gated TCN-GRU encoder + Transformer decoder (same backbone as the previous study), CV anchor
    data        GPU-resident frame store, window batching, observation-only normalization, physical loss
    train       centralized and federated training (baselines, Fed-SSBR = CQS + UDR, FedTR, and their ablations)
    jobs        experiment stages (FedTR: pilot / tune / main / ablation / robust; Fed-SSBR: ssbr-*)
"""
