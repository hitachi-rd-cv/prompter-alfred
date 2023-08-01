# Model Evaluation

## Inference

1. Create a Docker container and enter the container. Source code should be located in `/home/prompter` directory.

   ```bash
   docker-compose up -d
   docker exec -it {CONTAINER_NAME} bash
   cd /home/prompter
   ```

2. Run the evaluation

   - Using `main.py`

     ```bash
     python main.py -n1 --max_episode_length 1000 --num_local_steps 25 --num_processes 1 --eval_split valid_unseen --from_idx 0 --to_idx 510 --max_fails 10 --debug_local --learned_depth --use_sem_seg --set_dn testrun -v 0 --which_gpu 0 --x_display 0 --sem_policy_type mlm --mlm_fname mlmscore_equal --mlm_options aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay --seed 1 --splits alfred_data_small/splits/oct21.json --grid_sz 240 --mlm_temperature 1 --approx_last_action_success --language_granularity high --centering_strategy local_adjustment --target_offset_interaction 0.5 --obstacle_selem 9 --debug_env
     ```

   - Using `scripts/inference.sh` (details [here](scripts))

     ```bash
     ./scripts/inference_base.sh 0 510 1 valid_unseen testrun mlm mlmscore_equal "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" 1 240 1 high local_adjustment 0.5 9
     ```

   - **Caveat**: Multiprocessing (using --num_processes > 1) will make the construction of semantic mapping slower. We recommend that you use "--num_processes 2" (or a number around 2) and just run several jobs. (E.g. one job with episodes from 0 to 200, another job with episodes from 200 to 400, etc)



## List of Arguments for `main.py`

| Name                                          | Data type or candidates                                   | Recommended value | Descriptions                                                 |
| --------------------------------------------- | --------------------------------------------------------- | ----------------- | ------------------------------------------------------------ |
| approx_last_action_success                    | bool                                                      | True              | Estimate the action success.                                 |
| camera_height                                 | float                                                     | 1.576             | Height of the agent camera.                                  |
| centering_strategy                            | `local_adjustment,` `simple`, or `none`                   | local_adjustment  | Chooses which Local Adjustment Module to use. "local_adjustment": bases decisions on 3D target location, "simple": bases decisions on 2D target location (basically the same as FILM), "none": no target centering. |
| collision_obstacle_length                     | int                                                       | 9                 | The size of the obstacle added to the obstacle map when agent collides into something (in pixels). Note that 1 pixel = 5cm if `grid_sz`=240. |
| debug_env                                     |                                                           |                   | Adding this allows for break pointing (pdb.set_trace()) in sem_exp_thor.py.<br />Great for debugging your model. |
| debug_local                                   |                                                           |                   | Debug argument that will print statements that can help debugging. |
| eval_split                                    | `tests_seen,` `tests_unseen`, `val_seen`, or `val_unseen` |                   | Which evaluation split to run the model on.                  |
| from_idx                                      | int                                                       |                   | The index of episode to start in the "eval_split".           |
| goal_search_del_size                          | int                                                       | 11                | Size to remove from observation map after failed search      |
| grid_sz                                       | int                                                       | 240               | Granularity of the semantic search module prediction. Should be set to 8 for sem_policy_type = "cnn" |
| ignore_sliced                                 |                                                           |                   | Use the instance segmentation result of unsliced objects for sliced objects.<br />This is recommended, as the pretrained instance segmentation model cannot differentiate between sliced and unsliced objects. |
| language_granularity                          | `high`, `high_low`, or `gt`                               |                   | Granularity of the natural language instruction.             |
| learned_depth                                 |                                                           |                   | Use learned depth (ground truth depth used without it).      |
| manual_control                                |                                                           |                   | Overrides the system, the user can manually move the agent. More on this [here](debugging.md). |
| max_episode_length                            | int                                                       | 1000              | The episode automatically ends after this number of time steps. |
| max_fails                                     | int                                                       |                   | The episode automatically ends after this number of failed actions. |
| mlm_fname                                     | String                                                    |                   | Filename of the .csv file containing the collocation predictions by language prompting. |
| mlm_options                                   | String                                                    |                   | Miscellaneous options. Details in the next section.          |
| mlm_temperature                               | float                                                     |                   | Softmax temperature value used during collocation probability map sampling. |
| num_processes                                 | int                                                       |                   | Number of processes.                                         |
| obstacle_selem                                | int                                                       | 9                 | The amount of dilation applied to the obstacle prediction (in pixels). Note that 1 pixel = 5cm if `grid_sz`=240. |
| save_pictures                                 |                                                           |                   | Save the map, fmm_dist (visualization of fast marching method), RGB frame pictures. The pictures will be saved to "pictures/`args.eval_split`/`args.set_dn`" |
| seed                                          | int                                                       |                   | seed                                                         |
| sem_policy_type                               | `none`, `mlm`, `cnn`                                      | mlm               | Type of semantic search module used.<br />`none`: random search<br />`mlm`: language prompting<br />`cnn`: FILM's semantic search policy |
| sem_search_del_size                           | int                                                       | 11                | Size to remove from semantic search prediction after failed search |
| set_dn                                        | String                                                    |                   | Set the "name" of this run. The results will be saved in "/results/" under this name. Pictures will also be saved under this name with the --save_pictures flag. |
| target_offset_interaction                     | float                                                     | 0.5               | Agent offset from the obstacles, used during the OpenObject action. Units in meters. |
| to_idx                                        | int                                                       |                   | The index of episode to end in the "eval_split" (e.g. "--from_idx 0 --to_idx 1" will run the 0th episode). |
| use_sem_seg                                   |                                                           |                   | Use learned segmentation (ground truth segmentation used without it). |
| v                                             | 0 or 1                                                    | 0                 | Visualize (show windows of semantic map/ rgb on the monitor). **Do not use this on headless mode** |
| which_gpu, sem_gpu_id, sem_seg_gpu, depth_gpu | int                                                       |                   | Indices of gpus for semantic mapping, semantic search policy, semantic segmentation, depth. If you assign "--use_sem_seg --which_gpu 1 --sem_gpu_id 0 --sem_seg_gpu 0 --depth_gpu 1", gpu's of indices 0 and 1 will get almost equal loads for running 2 processes simultaneously. |
| x_display                                     | int                                                       |                   | Set this to the display number you have used for xserver (can use any number on a computer with a monitor). |



### `mlm_options`

Bolded options are used in the final submission.

| Name           | Descriptions |
| -------------- | ------------ |
| **sem_search_all** | Use the collocation probability map even if the next goal is not in the observed list. |
| spatial_norm<br />**aggregate_sum**<br />aggregate_max<br />aggregate_sample | Used [here](models/semantic_policy/sem_map_model.py)<br />Determines how LLM scores are aggregated to produce the collocation probability map. |
| **temperature_annealing** | Use temperature annealing when sampling from the collocation map. |
| **new_obstacle_fn** | An improved obstacle update scheme. |
| **no_slice_replay** | Disable slice replay (remember the location of all objects the agent has performed the `Slice` action). |
| no_lookaround | Disable looking around the environment at the start of the episode. |
| lookaroundAtSubgoal | When arriving at a subgoal, look around the environment |
| visibility_use_depth_distance | If this option is set, depth estimation is directly used to decide if an object is within reach, instead of converting the depth estimation in the world coordinate. |
| mixed_search | Also utilize the equal collocation probability map. |
| search_argmax_100 | When picking the next coordinate to visit from the collocation probability map, always pick the coordinate with the highest value. If this option is not set, the collocation map is sampled 50% of the time and max value is selected 50% of the time. |
| explored<br />past_loc | Used [here](models/semantic_policy/sem_map_model.py)<br />Determines explored or past location map to be used as the explored map. |



## Evaluating the Run

### Validation Set

Use [results/eval.py](results/eval.py). Make sure to change the script as follows:

1. Set `split` to be one of valid_unseen, valid_seen, test_unseen, or test_seen.
   - Note that ground truth SR calculation is only possible for the validation splits. Evaluation on test splits are internal/ approximate results.
2. Set `suffixes` with the name of the run.
3. Set `indx_range` and `result_tmpl` with how the run is split amongst different machines.



For example, let's say that you ran two experiments, **first_run** and **second_run**, on valid_unseen split, with following parameters for `main.py`.

| `--set_dn` | (`--from_idx`, `--to_idx`)       |
| ---------- | -------------------------------- |
| first_run  | (0, 300), (300, 600), (600, 821) |
| second_run | (0, 140), (140, 490), (490, 821) |

`(0, 300), (300, 600), (600, 821)` means that the run is split among 3 machines, first machine running experiments for scene #0-300, second machine running scene #300-600, and third machine running scene #600-821.

`eval.py` corresponding to this case would look something like this:

```python
if __name__ == '__main__':
    results_dname = "results"
    use_log = False
    suffixes = {"even": ["first_run"],
                "optimized": ["second_run"]}

    split = "valid_unseen"

    suffix_list = list()
    dfs, dfs_weighted, dfs_weighted_fail, path_lens = dict(), dict(), dict(), dict()
    log_info = dict()
    for indx_range, sfxs in suffixes.items():
        for suffix in sfxs:
            leaderboard_tmpl = None
            if indx_range == "even":
                result_tmpl = [
                    f'{split}_%s_0_to_300_{suffix}.p',
                    f'{split}_%s_300_to_600_{suffix}.p',
                    f'{split}_%s_600_to_821_{suffix}.p'
                ]
            elif indx_range == "optimized":
                result_tmpl = [
                    f'{split}_%s_0_to_140_{suffix}.p',
                    f'{split}_%s_140_to_490_{suffix}.p',
                    f'{split}_%s_490_to_821_{suffix}.p'
                ]
            ...
```

The keys of `suffixes` is used in the for loop to determine how the run was split.



### Test Set

Evaluating on test set requires submitting the run to the [Leaderboard](https://leaderboard.allenai.org/alfred/submissions/public). Run

```
$ python3 utils/leaderboard_script.py --dn_startswith WHAT_YOUR_PICKLES_START_WITH --json_name DESIRED_JSON_NAME
```

`utils/leaderboard_script.py` aggregates the results stored in `results/leaderboard/`, and outputs a json file in `leaderboard_jsons/`. Upload this to the leaderboard.