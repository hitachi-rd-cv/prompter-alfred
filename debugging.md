# Debugging Tips

### Setting breakpoints in the code

You can set `breakpoint()` in the code if you set `--debug_env` and `--num_processes=1`.



### Visualizing the model performance

If you want to visualize how the model is performing, you should first run the evaluation with `--save_pictures` flag. This will output frame-by-frame information in `pictures` directory.

If you want to combine the information and view it as a video, use the `visualize/viz.py` script. Refer to [README](visualize) for details.



### Manual control

You can move around in the environment using key inputs by setting `--manual_control` flag. You should also set `--save_pictures` to monitor how the scene is changing.

Refer to ```manualControl()``` in ```sem_exp_thor.py``` for details, and [here](scripts) for running `main.py` in manual control mode.

| Key input             | Action                                                       |
| --------------------- | ------------------------------------------------------------ |
| a                     | RotateLeft_90                                                |
| w                     | MoveAhead_25                                                 |
| d                     | RotateRight_90                                               |
| u                     | LookUp_15                                                    |
| n                     | LookDown_15                                                  |
| {Interaction},{X},{Y} | Perform {Interaction} at (X, Y) coordinate in the egocentric view. Example: ```OpenObject,320,20``` |
| thresh,{float}        | Change the threshold of the instance segmentation to {float}. Example: ```thresh,0.2``` |

