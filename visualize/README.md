# Visualization

If you want to visualize how the model is performing, you should first run the evaluation with `--save_pictures` flag. This will output frame-by-frame information in `pictures` directory.



[viz.py](viz.py) script can be used to compile the frame-by-frame information into a video. Make sure to change the script as follows:

1. Set `split` to be one of valid_unseen, valid_seen, test_unseen, or test_seen.
2. Set `run_name` with the name of the run.
3. Set `runs` with the list of the episode ID you want to create videos.
4. Set `run_range` as `{from_idx}_to_{to_idx}`. Values in `runs` must be within `from_idx` and `to_idx`.



For example, let's say that you ran an experiment, **testrun**, on the valid_unseen split, where `--from_idx=0` and `--to_idx=300` were set as parameters of the main.py.

`viz.py` to create a video for episodes 1, 2, and 10 would look something like this:

```python
def main():
    split_name = "valid_unseen"
    run_range = "0_to_300"
    run_name = "testrun"
    runs = [1, 2, 10]
    ...
```

The output .avi file should appear in the `visualize` directory.

