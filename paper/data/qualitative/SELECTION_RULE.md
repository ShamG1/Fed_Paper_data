# Selection rule for the trajectory examples

Fixed on 2026-10-25, before any per-window prediction of the checkpoints was computed. No window is replaced,
added, or dropped after the predictions are known; every selected window is shown.

Purpose: show single test windows of the hard strata without choosing them by how well CQS does on them.

- Settings: NGSIM-natural and highD-motion, the two main benchmarks.
- Models: the validation-selected checkpoints (`best.pt`) of seed 1 for FedAvg, CQS, and CQS+FedNova, and
  constant-velocity (CV) extrapolation as the kinematic reference.
- Strata on the test split, with the definitions of Fig. 4 and of `fedtraj/data.py: stratify`:
  - Q5: windows whose squared CV error at 5 s (mean over the two coordinates) lies above the 80th percentile of
    the test split;
  - LC (lane change): windows whose absolute lateral displacement at 5 s exceeds 3 m.
  A window can belong to both strata.
- Statistic: delta = e(CQS) - e(FedAvg), where e is the Euclidean distance between the predicted and the true
  position at 5 s, in meters. Negative delta means that CQS is closer to the truth.
- Selection: in each setting and stratum, the windows whose delta is nearest to the 10th, 50th, and 90th
  percentiles of delta over the stratum (`numpy.percentile`, linear interpolation; ties go to the lower test
  index). Each stratum thus contributes one window where CQS is clearly better than FedAvg, one typical window,
  and one where CQS is clearly worse: 2 settings x 2 strata x 3 = 12 windows.
- Reported with the figure: the size of each stratum, the share of its windows with delta < 0, and its median
  delta (seed 1).
- Check before use: the per-window errors must reproduce, for each of the six checkpoints, the test RMSE at
  1-5 s, the Q5 RMSE, and the lane-change RMSE stored in its `result.json`.
