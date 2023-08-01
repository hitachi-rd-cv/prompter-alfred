# Helpful scripts for running evaluations

As `main.py` has many command line arguments, retyping it on terminal every time becomes cumbersome.

We prepared `inference_base.sh`, which can be used as a template for running  experiments.

Here are some samples of how it can be used:

```bash
# Our IROS submission configuration
./scripts/inference_base.sh 0 821 0 valid_unseen testrun mlm mlmscore_equal "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" 1 240 1 high local_adjustment 0.5 9

# Random collocation map
./scripts/inference_base.sh 0 821 0 valid_unseen random none mlmscore_equal "aggregate_sum spatial_norm sem_search_all new_obstacle_fn no_slice_replay" 1 240 1 high

# FILM
./scripts/inference_base.sh 0 821 0 valid_unseen cnn cnn mlmscore_equal "aggregate_sum spatial_norm sem_search_all new_obstacle_fn no_slice_replay" 1 240 1 high

# Ground truth language instruction for ablation studies
./scripts/inference_base.sh 0 821 0 valid_unseen testrun mlm mlmscore_equal "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" 1 240 1 gt local_adjustment 0.5 9

# Low level instruction for ablation studies
./scripts/inference_base.sh 0 821 0 valid_unseen testrun mlm mlmscore_equal "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" 1 240 1 low local_adjustment 0.5 9
```



## Other `inference_*sh` files

We prepared convenience scripts other than `inference_base.sh` for following cases.

| name                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `inference_debug.sh`      | For debugging. Can set `set_trace` and debug interactively.  |
| `inference_gtSemDepth.sh` | For ablation study. Use ground truth depth and instance segmentation. |
| `inference_manual.sh`     | For debugging. Manually run the agent.                       |
| `inference_pics.sh`       | Output various information (agent view, segmentation/depth estimation, etc.) during inference. |
