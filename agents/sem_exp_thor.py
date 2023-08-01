import os, sys
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')

import pickle, json
import copy
import string

import torch
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import skimage.morphology
import cv2
from PIL import Image

from envs.utils.fmm_planner import planNextMove
import envs.utils.pose as pu
import alfred_utils.gen.constants as constants
from alfred_utils.env.thor_env_code import ThorEnvCode
from models.instructions_processed_LP.ALFRED_task_helper import get_list_of_highlevel_actions, determine_consecutive_interx, get_arguments, get_arguments_test, read_test_dict
from models.segmentation.segmentation_helper import SemgnetationHelper
#from models.depth.depth_helper import DepthHelper
import utils.control_helper as CH

import envs.utils.depth_utils as du
import envs.utils.rotation_utils as ru


class Sem_Exp_Env_Agent_Thor(ThorEnvCode):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, scene_names, rank):
        self.fails_cur = 0

        self.args = args
        self.seed = self.args.seed
        episode_no = self.args.from_idx + rank
        self.local_rng = np.random.RandomState(self.seed + episode_no)

        super().__init__(args, rank)

        # initialize transform for RGB observations
        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        

        # initializations for planning:
        self.selem = skimage.morphology.square(self.args.obstacle_selem)
        self.flattened = pickle.load(open("miscellaneous/flattened.p", "rb"))
        
        self.last_three_sidesteps = [None]*3
        self.picked_up = False
        self.picked_up_mask = None
        self.sliced_mask = None
        self.sliced_pose = None
        
        self.transfer_cat = {'ButterKnife': 'Knife', "Knife":"ButterKnife"}
        
        self.scene_names = scene_names
        self.scene_pointer = 0
        
        self.obs = None
        self.steps = 0
        
        self.action_5_count = 0
        self.goal_visualize = None
        self.prev_goal = None

        self.reached_goal = False
        
        self.test_dict = read_test_dict(
            self.args.test, self.args.language_granularity, 'unseen' in self.args.eval_split)

        #Segmentation
        self.seg = SemgnetationHelper(self)

        self.do_log = self.args.debug_local

        #Depth

    
    def load_traj(self, scene_name):
        json_dir = 'alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
        traj_data = json.load(open(json_dir))
        return traj_data

    
    def load_initial_scene(self):
        self.side_step_order = 0
        self.rotate_before_side_step_count = 0
        self.fails_cur = 0
        self.put_rgb_mask = None
        self.pointer = 0

        self.prev_rgb = None
        self.prev_depth = None
        self.prev_seg = None

        self.steps_taken = 0
        self.goal_name = None
        self.steps = 0
        self.last_err = ""
        self.prev_number_action = None
        self.move_until_visible_order = 0
        self.consecutive_steps = False
        self.cat_equate_dict = {} #map "key" category to "value" category
        self.rotate_aftersidestep = None
        self.errs = []
        self.logs = []

        exclude = set(string.punctuation)

        self.broken_grid = []
        self.where_block = []
        self.remove_connections_to = None

        self.reached_goal = False
        
        self.action_5_count = 0
        self.prev_goal = None
        
        self.last_three_sidesteps = [None]*3
        self.picked_up = False
        self.picked_up_mask = None
        self.sliced_mask = None
        self.sliced_pose = None

        episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank        
        self.local_rng = np.random.RandomState(self.seed + episode_no)

        try:
            traj_data = self.load_traj(self.scene_names[self.scene_pointer]); r_idx = self.scene_names[self.scene_pointer]['repeat_idx']
            self.traj_data = traj_data; self.r_idx = r_idx

            self.picture_folder_name = "pictures/" + self.args.eval_split + "/"+ self.args.dn + "/" + str(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank) + "/"
            if self.args.save_pictures and not (self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank in self.args.skip_indices):
                os.makedirs(self.picture_folder_name)
                os.makedirs(self.picture_folder_name + "/fmm_dist")
                os.makedirs(self.picture_folder_name + "/obstacles_pre_dilation")
                os.makedirs(self.picture_folder_name + "/Sem")
                os.makedirs(self.picture_folder_name + "/Sem_Map")
                os.makedirs(self.picture_folder_name + "/Sem_Map_Target")
                os.makedirs(self.picture_folder_name + "/rgb")
                os.makedirs(self.picture_folder_name + "/depth")
                os.makedirs(self.picture_folder_name + "/depth_thresholded")

            task_type = get_arguments_test(self.test_dict, traj_data)[1]
            sliced = get_arguments_test(self.test_dict, traj_data)[-1]
            list_of_actions, categories_in_inst, second_object, caution_pointers = get_list_of_highlevel_actions(
                traj_data, self.test_dict, self.args.nonsliced)

            self.sliced = sliced
            self.caution_pointers = caution_pointers
            if self.args.no_caution_pointers:
                self.caution_pointers = []
            self.print_log("list of actions is ", list_of_actions)
            self.task_type = task_type
            self.second_object = second_object

            self.reset_total_cat_new(categories_in_inst)
            obs, info = self.setup_scene(traj_data, task_type, r_idx, self.args) 
            goal_name = list_of_actions[0][0]
            info = self.reset_goal(True, goal_name, None)

            # ai2thor ver. 2.1.0 only allows for 90 degrees left/right turns
            # lr_turn = 90
            # num_turns = 360 // lr_turn
            curr_angle = self.camera_horizon
            # self.lookaround_seq = [f"LookUp_{curr_angle}"] + [f"RotateLeft_{lr_turn}"] * (360 // lr_turn) + [f"LookDown_{curr_angle}"]
            # self.lookaround_seq = [f"RotateLeft_{lr_turn}"] * num_turns + [f"LookUp_{curr_angle}"] + [f"RotateLeft_{lr_turn}"] * num_turns + [f"LookDown_{curr_angle}"]
            self.lookaround_seq = [f"LookUp_{curr_angle}"] + [f"RotateLeft_90"] * 3 + [f"LookDown_{curr_angle}"] + [f"RotateRight_90"] * 2
            # self.lookaround_seq = [f"RotateLeft_{lr_turn}"] * (360 // lr_turn)

            if "no_lookaround" in self.args.mlm_options:
                self.lookaround_seq = []

            self.lookaround_counter = 0

            self.target_search_sequence = list()
            self.centering_actions = list()
            self.centering_history = list()
            self.force_move = False

            if task_type == 'look_at_obj_in_light':
                self.total_cat2idx['DeskLamp'] = self.total_cat2idx['FloorLamp']
                self.cat_equate_dict['DeskLamp'] = 'FloorLamp' #if DeskLamp is found, consider it as FloorLamp

            if sliced:
                self.total_cat2idx['ButterKnife'] = self.total_cat2idx['Knife']
                self.cat_equate_dict['ButterKnife'] = 'Knife' #if ButterKnife is found, consider it as Knife

            actions_dict = {'task_type': task_type, 'list_of_actions': list_of_actions, 'second_object': second_object, 'total_cat2idx': self.total_cat2idx, 'sliced':self.sliced}
            self.print_log('total cat2idx is ', self.total_cat2idx)

            self.actions_dict = actions_dict
        except:
            self.print_log("Scene pointers exceeded the number of all scenes, for env rank", self.rank)
            obs = np.zeros(self.obs.shape)
            info = self.info
            actions_dict = None

        self.seg.update_agent(self)
            
        return obs, info, actions_dict
    
    def load_next_scene(self, load):
        if load == True:
            self.scene_pointer += 1
            obs, info, actions_dict = self.load_initial_scene()
            return obs, info, actions_dict
        
        return self.obs, self.info, self.actions_dict
     
    def update_last_three_sidesteps(self, new_sidestep):
        self.last_three_sidesteps = self.last_three_sidesteps[:2]
        self.last_three_sidesteps = [new_sidestep] + self.last_three_sidesteps

    def lookDiscretizer(self, action, mask=None, do_postprocess=True):
        # LookDown/LookUp should be performed in increments of 15 degrees, according to https://github.com/askforalfred/alfred/issues/87
        # so discretize the actions by 15 degrees
        act, angle = self.splitAction(action)
        for _ in range(angle // 15 - 1):
            obs, rew, done, info, success, a, target_instance, err, api_action = self.va_interact_new(
                f"{act}_{15}", mask, False)
            if not success:
                # abort if a look action fails
                if do_postprocess:
                    obs, seg_print = self.preprocess_obs_success(success, obs)
                return obs, rew, done, info, success, a, target_instance, err, api_action

        return self.va_interact_new(f"{act}_{15}", mask, do_postprocess)

    def splitAction(self, action):
        tokens = action.split("_")

        if len(tokens) == 1:
            return (tokens[0], None)

        act, num = tokens
        num = int(float(num))
        return act, num

    def va_interact_new(self, action, mask=None, do_postprocess=True):
        if ("Look" in action) and (self.splitAction(action)[-1] > 15):
            return self.lookDiscretizer(action, mask, do_postprocess)

        if "Angle" in action:
            return self.set_back_to_angle(self.splitAction(action)[-1])

        self.print_log(f"action taken in step {self.steps_taken}: {action}")
        self.last_action_ogn = action

        obs, rew, done, info, success, a, target_instance, err, api_action = \
                                super().va_interact(action, mask)

        if self.args.save_pictures:
            cv2.imwrite(
                self.picture_folder_name + "rgb/"+ "rgb_" + str(self.steps_taken) + ".png",
                obs[:3, :, :].transpose((1, 2, 0)))

        if not(success):
            self.fails_cur +=1

        if self.args.approx_last_action_success:
            success = CH._get_approximate_success(self.prev_rgb, self.event.frame, action)

        self.last_success = success

        #Use api action just for leaderboard submission purposes, as in https://github.com/askforalfred/alfred/blob/master/models/eval/leaderboard.py#L101
        self.actions = CH._append_to_actseq(success, self.actions, api_action)
        self.seg.update_agent(self)

        if do_postprocess:
            obs, seg_print = self.preprocess_obs_success(success, obs)

        return obs, rew, done, info, success, a, target_instance, err, api_action

    def set_back_to_angle(self, angle_arg):
        delta_angle = self.camera_horizon - angle_arg
        direction = "Up" if delta_angle >= 0 else "Down"
        delta_angle = abs(int(np.round(delta_angle / 15)) * 15)  # round to the nearest multiple of 15
        action = f"Look{direction}_{delta_angle}"

        return self.va_interact_new(action)

    def reset_goal(self, truefalse, goal_name, consecutive_interaction):
        if self.args.ignore_sliced:
            goal_name = goal_name.replace('Sliced', '')

        if truefalse == True:
            self.goal_name = goal_name
            if "Sliced" in goal_name :
                self.cur_goal_sliced = self.total_cat2idx[goal_name.replace('Sliced', '')]
            else:
                self.cur_goal_sliced = None
            self.goal_idx = self.total_cat2idx[goal_name]
            self.info['goal_cat_id'] = self.goal_idx
            self.info['goal_name'] = self.goal_name

            self.prev_number_action = None
            self.where_block = []

            self.search_end = False

            self.cur_goal_sem_seg_threshold_small = self.args.sem_seg_threshold_small
            self.cur_goal_sem_seg_threshold_large = self.args.sem_seg_threshold_large
            
            if abs(int(self.camera_horizon)  - 45) >5 and consecutive_interaction is None:
                obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(45)
            
            self.info['view_angle'] = self.camera_horizon
        
        return self.info
        
    def reset_total_cat_new(self, categories_in_inst):
        total_cat2idx = {}

        total_cat2idx["Knife"] =  len(total_cat2idx)
        total_cat2idx["SinkBasin"] =  len(total_cat2idx)
        if self.args.sem_policy_type != "none":
            for obj in constants.map_save_large_objects:
                if not(obj == "SinkBasin"):
                    total_cat2idx[obj] = len(total_cat2idx)


        start_idx = len(total_cat2idx)  # 1 for "fake"
        start_idx += 4 *self.rank
        cat_counter = 0
        assert len(categories_in_inst) <=6
        #Keep total_cat2idx just for 
        for v in categories_in_inst:
            if not(v in total_cat2idx):
                total_cat2idx[v] = start_idx+ cat_counter
                cat_counter +=1 
        
        total_cat2idx["None"] = 1 + 1 + 5 * self.args.num_processes-1
        if self.args.sem_policy_type != "none":
            total_cat2idx["None"] = total_cat2idx["None"] + 23
        self.total_cat2idx = total_cat2idx
        self.goal_idx2cat = {v:k for k, v in self.total_cat2idx.items()}
        print("self.goal_idx2cat is ", self.goal_idx2cat)
        self.cat_list = categories_in_inst
        self.args.num_sem_categories = 1 + 1 + 1 + 5 * self.args.num_processes 
        if self.args.sem_policy_type != "none":
            self.args.num_sem_categories = self.args.num_sem_categories + 23

    def setup_scene(self, traj_data, task_type, r_idx, args, reward_type='dense'):
        args = self.args

        obs, info = super().setup_scene(traj_data,task_type, r_idx, args, reward_type)
        obs, seg_print = self._preprocess_obs(obs)

        self.obs_shape = obs.shape
        self.obs = obs

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.col_width = 5
        self.curr_loc = [args.map_size_cm/100.0/2.0,
                         args.map_size_cm/100.0/2.0, 0.]
        self.last_action_ogn = None
        self.seg_print = seg_print

        return obs, info

    def getWorldCoord3DObs(self, depth_map):
        hei, wid = depth_map.shape[:2]
        cam_mat = du.get_camera_matrix(wid, hei, self.args.hfov)

        pcloud = du.get_point_cloud_from_z(depth_map, cam_mat)
        agent_view = du.transform_camera_view(pcloud, self.args.camera_height, -self.camera_horizon)

        scale = 100.0 / self.args.map_resolution
        shift_loc = [self.curr_loc[0] * scale, self.curr_loc[1] * scale, np.deg2rad(self.curr_loc[2])]
        global_view = du.transform_pose(agent_view * scale, shift_loc)

        return agent_view, global_view

    def is_visible_from_mask(self, mask, visibility_threshold=1):
        # use bottom 25 percentile for a more robust depth estimation
        # use distance horizontal to ground (and not distance from the camera) to check if it is visible
        if mask is None or np.sum(mask) == 0:
            return None, False

        depth_map = self.learned_depth_frame if self.args.learned_visibility else self.event.depth_frame / 1000

        # xyz is agent centric, units in meters
        # +x axis: right, +y axis: away from the agent, +z axis: up
        xyz, global_xyz = self.getWorldCoord3DObs(depth_map)

        mask_coords = np.where(mask)
        xyz_roi = xyz[mask_coords[0], mask_coords[1], :]
        global_xyz_roi = global_xyz[mask_coords[0], mask_coords[1], :]
        dists = np.sqrt(xyz_roi[:, 0] ** 2 + xyz_roi[:, 1] ** 2)
        gnd_dist = np.percentile(dists, 25)
        rep_indx = np.argmin(np.abs(dists - gnd_dist))

        reachable = gnd_dist <= visibility_threshold
        self.print_log(f"object is {gnd_dist}m away, reachable: {reachable}")
        return [int(coord) for coord in global_xyz_roi[rep_indx]], reachable

    def is_visible_from_mask_depth(self, mask, visibility_threshold=1):
        # for ablation studies
        # use bottom 25 percentile for a more robust depth estimation
        # use depth to check if it is visible
        if mask is None or np.sum(mask) == 0:
            return None, False

        depth_map = self.learned_depth_frame if self.args.learned_visibility else self.event.depth_frame / 1000

        # xyz is agent centric, units in meters
        # +x axis: right, +y axis: away from the agent, +z axis: up
        _, global_xyz = self.getWorldCoord3DObs(depth_map)

        mask_coords = np.where(mask)
        global_xyz_roi = global_xyz[mask_coords[0], mask_coords[1], :]

        dists = depth_map[mask_coords[0], mask_coords[1]]
        gnd_dist = np.percentile(dists, 25)
        rep_indx = np.argmin(np.abs(dists - gnd_dist))

        reachable = gnd_dist <= visibility_threshold
        self.print_log(f"object is {gnd_dist}m away, reachable: {reachable}")
        return [int(coord) for coord in global_xyz_roi[rep_indx]], reachable

    def preprocess_obs_success(self, success, obs):
        obs, seg_print = self._preprocess_obs(obs) #= obs, seg_print
        self.obs = obs
        self.seg_print = seg_print
        return obs, seg_print

    def genPickedUpMask(self, rgb, prev_rgb, original_seg):
        if original_seg is None:
            h, w = rgb.shape[:2]
            return np.zeros((h, w), np.uint8)

        # determine the location and shape of the picked up object by
        # checking which pixels have changed from the previous frame
        diff_thresh = 5
        bgr_s = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1]
        oldbgr_s = cv2.cvtColor(prev_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1]
        diff_mask = cv2.absdiff(bgr_s, oldbgr_s) > diff_thresh

        # diff_mask contains the location of the picked-up object, before and after the pickup action
        # so remove the mask before pickup
        # TODO: this fails for pixels that belong to the picked up object, before and after picking up
        original_seg = cv2.dilate(original_seg, np.ones((5, 5)))
        cleaned_mask = np.logical_and(diff_mask, np.logical_not(original_seg))
        cleaned_mask = cv2.dilate(cleaned_mask.astype(np.uint8), np.ones((5, 5)))

        return cleaned_mask

    def goBackToLastSlicedLoc(self, interaction):
        if "no_slice_replay" in self.args.mlm_options:
            return False, False

        pickup_sliced = (interaction == "PickupObject") and ("Sliced" in self.goal_idx2cat[self.goal_idx])
        on_path_back = pickup_sliced and (self.sliced_pose is not None)
        ready_to_slice = pickup_sliced and (not on_path_back) and (self.sliced_mask is not None)
        return on_path_back, ready_to_slice

    def consecutive_interaction(self, interaction, interaction_mask):
        if interaction == "PutObject" and self.last_action_ogn == "OpenObject":
            interaction_mask = self.open_mask
        elif interaction == "CloseObject":
            interaction_mask = self.open_mask

        obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(
            interaction, interaction_mask, do_postprocess=False)

        if interaction == "PickupObject":
            if not success:
                interaction_mask = self.put_rgb_mask
                obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(
                    interaction, interaction_mask, do_postprocess=False)

            if success:
                self.picked_up = True
                self.picked_up_mask = self.genPickedUpMask(
                    self.event.frame, self.prev_rgb, interaction_mask)

        elif interaction == "PutObject" and success:
            self.picked_up = False

            self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.event.frame)
            self.picked_up_mask = None

        # update the depth information after picked_up-related information are updated
        obs, seg_print = self.preprocess_obs_success(success, obs)

        # store the current interact mask in anticipation to use it in the future
        if success:
            if interaction == "OpenObject":
                self.open_mask = copy.deepcopy(interaction_mask)
            elif interaction == "SliceObject":
                self.sliced_mask = copy.deepcopy(interaction_mask)
                self.sliced_pose = [self.curr_loc_grid[0], self.curr_loc_grid[1], self.curr_loc[2]]

        self.info = info

        return obs, rew, done, info, success, err

    def which_direction(self, interaction_mask):
        if interaction_mask is None:
            return 150
        widths = np.where(interaction_mask !=0)[1]
        center = np.mean(widths)
        return center

    def isTrapped(self, traversible, planning_window, area_thresh=400):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            traversible.astype(np.uint8), connectivity=4)

        [gx1, gx2, gy1, gy2] = planning_window
        agent_loc = self.meter2coord(self.curr_loc[1], self.curr_loc[0], gy1, gx1, traversible.shape)
        roi_label = labels[agent_loc[0], agent_loc[1]]
        area = stats[roi_label][4]
        trapped = area < area_thresh
        if trapped:
            self.print_log(f"Current area is too small (area: {area}), I think I'm trapped!")
        return trapped

    def get_traversible_new(self, grid, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        obstacles = grid[y1:y2, x1:x2]

        # add collision map
        collision_map = self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1

        # dilate the obstacles so that the agent doesn't get too close to the obstacles
        # but do not dilate the obstacles added via collision_map
        obstacles = skimage.morphology.binary_dilation(obstacles, self.selem)
        obstacles[collision_map] = 1

        # remove visited path from the obstacles
        obstacles[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        traversible = np.logical_not(obstacles)

        # check if the agent is trapped
        # if the agent is trapped, only use the collision_map as the obstacle map
        if self.isTrapped(traversible, planning_window):
            obstacles = np.zeros_like(obstacles)
            obstacles[collision_map] = 1
            traversible = np.logical_not(obstacles)

        return traversible

    def get_traversible(self, grid, planning_window):
        if "new_obstacle_fn" in self.args.mlm_options:
            return self.get_traversible_new(grid, planning_window)

        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        obstacles = grid[y1:y2, x1:x2]

        # add collision map
        collision_map = self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1

        # dilate the obstacles so that the agent doesn't get too close to the obstacles
        # but do not dilate the obstacles added via collision_map
        obstacles = skimage.morphology.binary_dilation(obstacles, self.selem)
        obstacles[collision_map] = 1

        # remove visited path from the obstacles
        obstacles[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        traversible = np.logical_not(obstacles)

        # check if the agent is trapped
        # if the agent is trapped, only use the collision_map as the obstacle map
        if self.isTrapped(traversible, planning_window):
            obstacles = np.zeros_like(obstacles)
            obstacles[collision_map] = 1
            traversible = np.logical_not(obstacles)

        return traversible

    def getGoalDirection(self, planner_inputs):
        agent_x, agent_y, curr_angle, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
        agent_x = int(agent_x * 100.0/self.args.map_resolution - gx1)
        agent_y = int(agent_y * 100.0/self.args.map_resolution - gy1)

        ys, xs = np.where(planner_inputs["goal"])

        distances = (xs - agent_x) ** 2 + (ys - agent_y) ** 2
        mindx = np.argmin(distances)
        target_x, target_y = xs[mindx], ys[mindx]

        dx = target_x - agent_x
        dy = target_y - agent_y

        if abs(dx) > abs(dy):
            if dx > 0:
                target_angle = 0
            else:
                target_angle = 180
        else:
            if dy > 0:
                target_angle = 90
            else:
                target_angle = 270

        delta_angle = (target_angle - curr_angle) % 360
        return delta_angle

    def arrivedAtSubGoal(self, planner_inputs, goal_free_and_not_shifted):
        self.print_log("Arrived at subgoal")

        if goal_free_and_not_shifted or ("lookaroundAtSubgoal" in self.args.mlm_options):
            self.print_log("Looking in all 4 directions")
            return ["RotateLeft_90"] * 3 + ["Angle_0"] + ["RotateLeft_90"] * 3 + ["Angle_45", "Done"]

        self.print_log("Looking only in 1 direction")
        actions = list()

        cur_hor = np.round(self.camera_horizon, 4)
        if abs(cur_hor-45) > 5:
            actions.append("Angle_45")

        # figure out which direction the agent must turn to
        delta_angle = self.getGoalDirection(planner_inputs)
        if delta_angle == 0:  # no need to turn
            pass
        elif delta_angle == 90:
            actions += ["RotateLeft_90"]
        elif delta_angle == 270:
            actions += ["RotateRight_90"]
        elif delta_angle == 180:
            # rotate left twice to do a 180 degrees turn
            actions += ["RotateLeft_90", "RotateLeft_90"]

        # the agent is looking at the correct direction, so look up and down
        # NOTE: last item in actions needs to be "Done", since that is how getNextAction() knows
        # if search sequence failed
        actions += ["LookUp_0", "Angle_0", "Angle_45", "Done"]

        return actions

    def getSceneNum(self):
        return self.scene_names[self.scene_pointer]["scene_num"]

    def centerAgentNone(self, planner_inputs, interaction_mask):
        # no agent centering, for ablation study
        self.print_log("agent centered")
        return True, False

    def centerAgentSimple(self, planner_inputs, interaction_mask):
        # 2D agent centering in FILM, for ablation study

        # figure out which direction to move and generate a sequence of actions to center the agent
        self.print_log("check centered")

        self.centering_actions = list()
        wd = self.which_direction(interaction_mask)

        width = interaction_mask.shape[1]
        margin = 65
        if wd > (width - margin):
            self.print_log(f"stepping to right, wd={wd}")
            turn_angle = -90
            self.centering_actions = ["RotateRight_90", "MoveAhead_25", "RotateLeft_90"]
            self.centering_history.append('R')

        elif wd < margin:
            self.print_log(f"stepping to left, wd={wd}")
            turn_angle = 90
            self.centering_actions = ["RotateLeft_90", "MoveAhead_25", "RotateRight_90"]
            self.centering_history.append('L')

        else:
            self.print_log("agent centered")
            return True, False

        # check for collisions
        grid = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        start = [int(start_y * 100.0/self.args.map_resolution - gx1),
                 int(start_x * 100.0/self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)

        traversible = self.get_traversible(grid, planning_window)

        turn_angle = start_o + turn_angle
        movable = CH._check_five_pixels_ahead_map_pred_for_moving(self.args, traversible, start, turn_angle)
        if not movable:
            # do not include MoveAhead, since the agent will collide
            self.centering_actions = [self.centering_actions[0]]

        # check for repeated left and right movement
        # assumption here is that the agent should always move in one direction
        repeating_steps = len(set(self.centering_history)) > 1
        if repeating_steps:
            self.print_log("adjust movement repetition detected")
            self.centering_history = list()
            self.centering_actions = list()

            self.print_log(f"aborting centering sequence")
        abort_centering = repeating_steps

        return False, abort_centering

    def centerAgent(self, planner_inputs, interaction_mask):
        # returns True if agent is centered, else it sets centering_actions for next actions

        def searchTargetPt(depth_map, camera_matrix, target_pt, search_range=2):
            # target_pt should be ordered (y, x)

            # search for a coordinate near target_pt with a reliable depth prediction
            reliable_pt = None
            h, w = depth_map.shape[:2]
            pty, ptx = target_pt
            for i in range(search_range):
                lower_y, upper_y = max(0, pty - i), min(h, pty + i + 1)
                lower_x, upper_x = max(0, ptx - i), min(w, ptx + i + 1)
                ys, xs = np.where(depth_map[lower_y:upper_y, lower_x:upper_x])
                if len(xs) > 0:
                    reliable_pt = (lower_y + ys[0], lower_x + xs[0])
                    break

            if reliable_pt is None:
                return None

            xyz = du.get_point_cloud_from_z(depth_map, camera_matrix)
            return xyz[reliable_pt[0], reliable_pt[1], :]

        # figure out which direction to move and generate a sequence of actions to center the agent
        self.print_log("check centered")

        self.centering_actions = list()
        wd = self.which_direction(interaction_mask)

        # rotation matrix and translation vector are defined from the current coordinate system,
        # where +x is right, +y is the direction in which the agent is facing, and +z is up, in accordance with depth_utils
        Rmat = np.eye(3)

        # left and right commands for agent are with respect to the world coordinate, but
        # the agent could be looking up/downward, making the camera coordinate different from the world coordinate,
        # so correct it
        x_axis = np.asarray([1, 0, 0])
        z_axis = np.asarray([0, 0, 1])
        head_angle = -self.camera_horizon  # camera_horizon is defined as negative rotation around the x axis of the camera coordinate
        correction_R = ru.get_r_matrix(x_axis, -head_angle / 180 * np.pi)  # undo rotation by @head_angle
        world_z_vec = correction_R @ z_axis

        width = interaction_mask.shape[1]
        if wd > (width - self.center_margin):
            self.print_log(f"stepping to right, wd={wd}")
            turn_angle = -90
            self.centering_actions = ["RotateRight_90", "MoveAhead_25", "RotateLeft_90"]
            self.centering_history.append('R')

            Rmat_tmp = ru.get_r_matrix(world_z_vec, np.pi / 2)
            tvec = np.asarray([-25, 0, 0])  # depth values in obs is in centimeters

        elif wd < self.center_margin:
            self.print_log(f"stepping to left, wd={wd}")
            turn_angle = 90
            self.centering_actions = ["RotateLeft_90", "MoveAhead_25", "RotateRight_90"]
            self.centering_history.append('L')

            Rmat_tmp = ru.get_r_matrix(world_z_vec, -np.pi / 2)
            tvec = np.asarray([25, 0, 0])

        else:
            self.print_log("agent centered")
            return True, False

        # check for collisions
        grid = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        start = [int(start_y * 100.0/self.args.map_resolution - gx1),
                 int(start_x * 100.0/self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)

        traversible = self.get_traversible(grid, planning_window)

        turn_angle = start_o + turn_angle
        movable = CH._check_five_pixels_ahead_map_pred_for_moving(self.args, traversible, start, turn_angle)
        if not movable:
            # do not include MoveAhead, since the agent will collide
            self.centering_actions = [self.centering_actions[0]]
            Rmat = Rmat_tmp
            tvec = np.zeros_like(tvec)

        # check for repeated left and right movement
        # assumption here is that the agent should always move in one direction
        repeating_steps = len(set(self.centering_history)) > 1
        if repeating_steps:
            self.print_log("adjust movement repetition detected")

        # calculate where the target coordinate is, after all actions in centering_actions are executed
        # 1. calculate the target coordinate and find its 3D coordinate
        imask_h, imask_w = interaction_mask.shape[:2]
        ratio = self.args.frame_height / interaction_mask.shape[0]
        curr_target_pt = np.asarray([np.average(indices) * ratio for indices in np.where(interaction_mask)], dtype=int)

        # search for coordinates near curr_target_pt with a reliable depth prediction
        cam_mat = du.get_camera_matrix(self.args.frame_width, self.args.frame_height, self.args.hfov)
        curr_target_pt_3d = searchTargetPt(self.obs[3, :, :], cam_mat, curr_target_pt, search_range=3)
        if curr_target_pt_3d is None:
            proj_pt = (-1, -1)
        else:
            # 2. project the 3D coordinate to future camera coordinate
            proj_pt = du.projectPoints(curr_target_pt_3d, Rmat, tvec, cam_mat.array, self.args.frame_height) / ratio

        self.print_log(f"expects target at {curr_target_pt / ratio} to move to {proj_pt} after adjustment")
        self.centering_actions += ["Done", proj_pt]

        # abort if the expected coordinate is out of visual bounds
        target_oob = (proj_pt[0] < 0) or (proj_pt[1] < 0) or (proj_pt[0] >= imask_h) or (proj_pt[1] >= imask_w)
        if target_oob:
            self.print_log("tracking target out of bounds")

        abort_centering = repeating_steps | target_oob
        if abort_centering:
            self.centering_history = list()
            self.centering_actions = list()
            abort_centering = True

            self.center_margin = max(0, self.center_margin - 10)
            self.print_log(f"aborting centering sequence, margin reduced to {self.center_margin}")

        return False, abort_centering

    def interactionSequence(self, planner_inputs, interaction_mask):
        # interact with the target
        list_of_actions = planner_inputs['list_of_actions']
        pointer = planner_inputs['list_of_actions_pointer']
        interaction = list_of_actions[pointer][1]

        obs, rew, done, info, success, err = self.consecutive_interaction(
            interaction, interaction_mask)
        self.force_move = not success

        return obs, rew, done, info, success, err

    def getNextAction(self, planner_inputs, during_lookaround, target_offset, target_xyz, force_slice_pickup):
        next_goal = False
        reached_goal = False

        # lookaround sequence
        if during_lookaround:
            self.print_log("action chosen: lookaround")
            # initial lookaround sequence
            action = self.lookaround_seq[self.lookaround_counter]
            self.lookaround_counter += 1
            next_goal = len(self.lookaround_seq) == self.lookaround_counter

        # if sliced mask is available and now want to pick up the sliced object,
        # go back to the location in which the agent sliced last time
        elif force_slice_pickup:
            self.print_log("action chosen: force_slice_pickup, overriding goal")
            action, _, goal_free_and_not_shifted = self._plan(
                planner_inputs, target_offset, self.sliced_pose)

            # upon arrival, try to pick up the sliced object
            if action in ["ReachedSubgoal", "<<stop>>"]:
                action = "SliceObjectFromMemory"

        # # can see the target, but not close enough
        # elif target_xyz is not None:
        #     self.print_log("action chosen: target_visible")
        #     mod_goal = np.zeros_like(planner_inputs["goal"])
        #     mod_goal[target_xyz[1], target_xyz[0]] = 1
        #     action, next_subgoal = self._plan(planner_inputs, target_offset, mod_goal, False)

        # arrived at the subgoal, search sequence
        elif len(self.target_search_sequence) != 0:
            self.print_log("action chosen: target search")
            action = self.target_search_sequence.pop(0)
            reached_goal = True
            if self.target_search_sequence[0] == "Done":
                next_goal = True
                self.target_search_sequence = list()

        # check to make sure that the agent is looking at 45 deg below horizon
        elif self.camera_horizon != 45:
            self.print_log("action chosen: correct agent angle to 45")
            action = "Angle_45"

        else:
            self.print_log("action chosen: _plan")
            goal_coord = np.where(planner_inputs['goal'])
            goal_coord = [goal_coord[0][0], goal_coord[1][0]]
            action, next_goal, goal_free_and_not_shifted = self._plan(
                planner_inputs, target_offset, goal_coord)

        # found the target object and arrived at the location
        if action in ["ReachedSubgoal", "<<stop>>"]:
            self.target_search_sequence = self.arrivedAtSubGoal(planner_inputs, goal_free_and_not_shifted)
            action = self.target_search_sequence.pop(0)
            reached_goal = True

        return action, next_goal, reached_goal

    def check4interactability(self, target_coord, visibility_threshold):
        if self.args.use_sem_seg:
            interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(
                self.goal_idx2cat[self.goal_idx], target_coord, self.args.ignore_sliced)
        else:
            interaction_mask = self.seg.get_instance_mask_from_obj_type(
                self.goal_idx2cat[self.goal_idx], target_coord)

        if "visibility_use_depth_distance" in self.args.mlm_options:
            target_loc, reachable = self.is_visible_from_mask_depth(interaction_mask, visibility_threshold)
        else:
            target_loc, reachable = self.is_visible_from_mask(interaction_mask, visibility_threshold)

        # guess the interaction mask and force interaction if the target tracked during centering sequence is lost
        target_lost = (target_coord is not None) and ((interaction_mask is None) or (not reachable))
        if target_lost:
            self.print_log(f"target expected at {target_coord} is lost, force interaction")
            interaction_mask = np.zeros((300, 300), dtype=np.float64)
            interaction_mask[int(target_coord[0]), int(target_coord[1])] = 1
            reachable = True

        return target_loc, reachable, interaction_mask

    def isNewGoal(self, curr_goal):
        is_same_goal = np.array_equal(self.prev_goal, curr_goal)
        self.prev_goal = curr_goal
        return not is_same_goal

    def interactionProcess(self, interaction_fn, needs_centering, planner_inputs, interaction_mask):
        action = None
        obs, rew, done, info, goal_success, err = None, None, None, None, False, None
        abort_centering = False
        if needs_centering:
            centering_fns = {"local_adjustment": self.centerAgent, "simple": self.centerAgentSimple, "none": self.centerAgentNone}
            centering_fn = centering_fns[self.args.centering_strategy]
            done_centering, abort_centering = centering_fn(planner_inputs, interaction_mask)

        if (not needs_centering) or done_centering:
            obs, rew, done, info, goal_success, err = interaction_fn()
        elif not abort_centering:
            action = self.centering_actions.pop(0)

        if not abort_centering:
            self.target_search_sequence = list()

        return obs, rew, done, info, goal_success, err, abort_centering, action

    def convertManualInput(self, code):
        ctrl_map = {'a': "RotateLeft_90", 'w': "MoveAhead_25", 'd': "RotateRight_90", 'u': 'LookUp_15', 'n': 'LookDown_15'}
        if code in ctrl_map:
            return ctrl_map[code], None

        if len(code.split(',')) != 3:
            return "LookUp_0", None

        action, coordx, coordy = code.split(',')
        interaction_mask = np.zeros((300, 300), dtype=np.float64)
        interaction_mask[int(coordy), int(coordx)] = 1

        return action, interaction_mask

    def specialManualInputs(self, code):
        codes = code.split(',')

        if (len(codes) == 2) and (codes[0] == "thresh"):
            thresh = float(codes[1])
            self.args.sem_seg_threshold_small = thresh
            self.args.sem_seg_threshold_large = thresh
        elif (len(codes) == 1) and (codes[0] == "show_all"):
            self.args.override_sem_seg_thresh = True
        else:
            return False

        return True

    def manualControl(self, code):
        if self.specialManualInputs(code):
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new("LookUp_0")

        else:
            action, mask = self.convertManualInput(code)
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(action, mask)

        next_step_dict = {
            'keep_consecutive': False, 'view_angle': self.camera_horizon,
            'picked_up': self.picked_up, 'errs': self.errs, 'steps_taken': self.steps_taken,
            'broken_grid':self.broken_grid,
            'actseq':{(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank, self.traj_data['task_id']): self.actions[:1000]}, 
            'logs':self.logs,  'current_goal_sliced':self.cur_goal_sliced, 'next_goal': False,
            'delete_lamp': False, 'fails_cur': self.fails_cur}

        return obs, rew, done, info, False, next_step_dict

    def plan_act_and_preprocess(self, planner_inputs, goal_spotted):
        self.fails_cur = 0
        self.pointer = planner_inputs['list_of_actions_pointer']
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) matrix denoting goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, and evaluation metric info
        """
        def updateVisited(pose_pred):
            curr_x, curr_y, curr_o, gx1, gx2, gy1, gy2 = pose_pred

            self.last_loc = self.curr_loc
            self.curr_loc = [curr_x, curr_y, curr_o]

            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            self.curr_loc_grid = self.meter2coord(
                self.curr_loc[1], self.curr_loc[0], gy1, gx1, self.visited.shape)
            prev = self.meter2coord(self.last_loc[1], self.last_loc[0], gy1, gx1, self.visited.shape)

            self.visited[gx1:gx2, gy1:gy2] = cv2.line(
                self.visited[gx1:gx2, gy1:gy2], (prev[1], prev[0]),
                (self.curr_loc_grid[1], self.curr_loc_grid[0]), 1, 1)

        self.steps += 1
        sdroate_direction = None
        next_goal = False
        abort_centering = False
        goal_success = False
        keep_consecutive = False
        action = None

        updateVisited(planner_inputs['pose_pred'])

        if self.isNewGoal(planner_inputs["goal"]):
            self.action_5_count = 0
            self.reached_goal = False
            self.print_log("newly goal set")
            self.center_margin = 65

        if planner_inputs["wait"]:
            self.last_action_ogn = None
            self.info["sensor_pose"] = [0., 0., 0.]
            self.rotate_aftersidestep =  None
            return np.zeros(self.obs.shape), 0., False, self.info, False, {
                'view_angle': self.camera_horizon,  'picked_up': self.picked_up,
                'steps_taken': self.steps_taken, 'broken_grid': self.broken_grid,
                'actseq':{(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank, self.traj_data['task_id']): self.actions[:1000]},
                'errs': self.errs, 'logs':self.logs, 'current_goal_sliced':self.cur_goal_sliced,
                'next_goal': next_goal, 'delete_lamp': False, 'fails_cur': 0}

        # check for collision
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
        self.collisionHandling(gx1, gy1)

        self._visualize(planner_inputs)

        # manual control by the user
        if planner_inputs["manual_step"] is not None:
            return self.manualControl(planner_inputs["manual_step"])

        # check if the target is within interactable distance,
        # but do not execute interaction during the initial lookaround sequence
        during_lookaround = self.lookaround_counter < len(self.lookaround_seq)

        # if the next action is in caution_pointers, adjust agent location to center the goal
        needs_centering = planner_inputs['list_of_actions_pointer'] in self.caution_pointers

        target_coord = None
        if (len(self.centering_actions) > 0) and (self.centering_actions[0] == "Done"):
            target_coord = self.centering_actions[-1]
            self.centering_actions = list()

        # AI2THOR agent can only see objects that are within 1.5m from itself
        # so set 1.45m to account for errors in depth estimation
        visibility_threshold = 1.45
        target_xyz, interactable, interaction_mask = self.check4interactability(target_coord, visibility_threshold)
        execute_interaction = interactable & (not during_lookaround)

        # if sliced mask is available and now want to pick up the sliced object,
        # do not interact with object along the path
        list_of_actions = planner_inputs['list_of_actions']
        pointer = planner_inputs['list_of_actions_pointer']
        interaction = list_of_actions[pointer][1]
        force_slice_pickup, _ = self.goBackToLastSlicedLoc(interaction)
        if force_slice_pickup:
            self.force_move = True

        # an interaction happens only when goal is spotted AND the agent is near enough to the goal
        # i.e. do not finish tasks during/along the path if it is a found goal
        interact_ok = (not self.force_move) and ((not needs_centering) or (not goal_spotted) or self.reached_goal)

        do_move = False
        is_centering = len(self.centering_actions) > 0
        if is_centering:
            self.print_log("centering_history", self.centering_history)

            high_priority_act = self.centering_actions.pop(0)
            obs, rew, done, info, success, err, _ = self.execAction(high_priority_act)
            self.print_log("action chosen: high priority")

        # run consecutive actions (ex. open -> close -> turn on -> turn off a microwave)
        elif interact_ok and planner_inputs['consecutive_interaction'] != None:
            self.print_log("action chosen: consecutive interaction")

            interaction_fn = lambda: self.consecutive_interaction(
                planner_inputs['consecutive_interaction'], interaction_mask)

            obs, rew, done, info, goal_success, err, abort_centering, action = self.interactionProcess(
                interaction_fn, needs_centering, planner_inputs, interaction_mask)

        # run a one-step interaction
        elif interact_ok and planner_inputs['consecutive_interaction'] == None and execute_interaction:
            self.print_log("action chosen: single interaction")

            interaction_fn = lambda: self.interactionSequence(planner_inputs, interaction_mask)

            obs, rew, done, info, goal_success, err, abort_centering, action = self.interactionProcess(
                interaction_fn, needs_centering, planner_inputs, interaction_mask)

        else:
            do_move = True

        if do_move or abort_centering:
            self.force_move = False
            goal_xyz = target_xyz if goal_spotted else None

            target_offset = (int(self.args.target_offset_interaction * 100.0/self.args.map_resolution)
                             if (interaction == "OpenObject") else 0)
            action, next_goal, self.reached_goal = self.getNextAction(
                planner_inputs, during_lookaround, target_offset, goal_xyz, force_slice_pickup)

        if action is not None:
            obs, rew, done, info, _, err, goal_success = self.execAction(action)

        will_center = len(self.centering_actions) > 0
        if (not is_centering) and (not will_center):
            self.centering_history = list()

        delete_lamp = (self.goal_name == 'FloorLamp') and (self.action_received == "ToggleObjectOn")
        if self.args.no_delete_lamp:
            delete_lamp = False

        # request next goal when non-movement action fails
        act, _ = self.splitAction(self.last_action_ogn)
        interaction_failed = (act not in ["MoveAhead", "RotateLeft", "RotateRight", "LookDown", "LookUp"]) and (not self.last_success)
        next_goal |= interaction_failed

        # request next goal when an interaction succeeds
        next_goal |= goal_success

        self.rotate_aftersidestep = sdroate_direction
        next_step_dict = {
            'keep_consecutive': keep_consecutive, 'view_angle': self.camera_horizon,
            'picked_up': self.picked_up, 'errs': self.errs, 'steps_taken': self.steps_taken,
            'broken_grid':self.broken_grid,
            'actseq':{(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank, self.traj_data['task_id']): self.actions[:1000]},
            'logs':self.logs,  'current_goal_sliced':self.cur_goal_sliced, 'next_goal': next_goal,
            'delete_lamp': delete_lamp, 'fails_cur': self.fails_cur}

        if err != "":
            self.print_log(f"step: {self.steps_taken} err is {err}")

        self.last_err = err
        self.info = info

        list_of_actions = planner_inputs['list_of_actions']
        pointer = planner_inputs['list_of_actions_pointer']
        if goal_success and (pointer + 1 < len(list_of_actions)):
            self.print_log("pointer increased goal name ", list_of_actions[pointer+1])

        return obs, rew, done, info, goal_success, next_step_dict

    def collisionHandling_new(self, gx1, gy1):
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        collision = (abs(x1 - x2) < self.args.collision_threshold) and (abs(y1 - y2) < self.args.collision_threshold)
        moved_ahead = self.last_action_ogn == "MoveAhead_25"

        if moved_ahead and collision:
            y, x = self.curr_loc_grid

            step_sz = self.args.step_size
            robot_radius = self.args.obstacle_selem // 2
            shift_sz = robot_radius + 1
            width = step_sz
            height = self.args.obstacle_selem

            rad = np.deg2rad(t1)
            rmat = np.asarray([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            for i in range(height):
                for j in range(shift_sz, shift_sz + width):
                    delta = np.asarray([j, i - self.args.collision_obstacle_length // 2])
                    dx, dy = rmat @ delta
                    [r, c] = pu.threshold_poses([int(y + dy), int(x + dx)], self.collision_map.shape)
                    self.collision_map[r, c] = 1

    def collisionHandling(self, gx1, gy1):
        if "new_obstacle_fn" in self.args.mlm_options:
            return self.collisionHandling_new(gx1, gy1)

        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        collision = (abs(x1 - x2) < self.args.collision_threshold) and (abs(y1 - y2) < self.args.collision_threshold)
        moved_ahead = self.last_action_ogn == "MoveAhead_25"

        if moved_ahead and collision:
            y, x = self.curr_loc_grid

            width = 3
            rad = np.deg2rad(t1)
            rmat = np.asarray([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            for i in range(self.args.collision_obstacle_length):
                for j in range(1, width+1):
                    delta = np.asarray([j, i - self.args.collision_obstacle_length // 2])
                    dx, dy = rmat @ delta
                    [r, c] = pu.threshold_poses([int(y + dy), int(x + dx)], self.collision_map.shape)
                    self.collision_map[r, c] = 1

    def execAction(self, action):
        goal_success = False
        if action == "SliceObjectFromMemory":
            obs, rew, done, info, success, err = self.consecutive_interaction(
                "PickupObject", self.sliced_mask)
            self.sliced_mask = None
            self.sliced_pose = None
            goal_success = success
        else:
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(action)

        self.print_log("obj type for mask is :", self.goal_idx2cat[self.goal_idx])

        return obs, rew, done, info, success, err, goal_success

    def meter2coord(self, y, x, miny, minx, map_shape):
        r, c = y, x
        start = [int(r * 100.0/self.args.map_resolution - minx),
                 int(c * 100.0/self.args.map_resolution - miny)]
        return pu.threshold_poses(start, map_shape)

    def _plan(self, planner_inputs, target_offset, goal_coord, measure_offset_from_edge=True):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        next_goal = False

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        start = self.curr_loc_grid
        start = [start[0], start[1], start_o]

        self.goal_visualize = np.zeros_like(planner_inputs["goal"])
        self.goal_visualize[goal_coord[0], goal_coord[1]] = 1
        nextAction, stop, next_goal, goal_free_and_not_shifted = self._get_stg(
            map_pred, start, goal_coord, planning_window, target_offset, measure_offset_from_edge)

        if nextAction is None:
            action = "LookUp_0"
        elif stop and planner_inputs["found_goal"]:
            action = "<<stop>>"
        elif stop:
            if self.action_5_count < 1:
                action = "ReachedSubgoal"
                self.action_5_count +=1
            else:
                next_goal = True
                action = "LookUp_0"
        else:
            action = nextAction

        return action, next_goal, goal_free_and_not_shifted

    def _get_stg(self, grid, start, goal_coord, planning_window, target_offset, measure_offset_from_edge=False):
        traversible = self.get_traversible(grid, planning_window)

        nextAction, new_goal, stop, next_goal, goal_free_and_not_shifted = planNextMove(
            traversible, self.args.step_size, start, goal_coord, target_offset, measure_offset_from_edge)

        if self.args.save_pictures:
            viz = np.repeat(255 - traversible[:, :, np.newaxis] * 255, 3, axis=2)
            h, w = viz.shape[:2]
            viz[max(0, goal_coord[0]-2):min(h-1, goal_coord[0]+3), max(0, goal_coord[1]-2):min(w-1, goal_coord[1]+3), :] = [255, 0, 0]
            viz[max(0, new_goal[0]-2):min(h-1, new_goal[0]+3), max(0, new_goal[1]-2):min(w-1, new_goal[1]+3), :] = [0, 0, 255]
            cv2.imwrite(self.picture_folder_name +"fmm_dist/"+ "fmm_dist_" + str(self.steps_taken) + ".png", viz)

            # save the obstacle images pre dilation for debugging purposes
            obst = grid.astype(np.uint8) * 255
            cv2.imwrite(self.picture_folder_name +"obstacles_pre_dilation/"+ "obst_" + str(self.steps_taken) + ".png", obst)

        return nextAction, stop, next_goal, goal_free_and_not_shifted

    def depth_pred_later(self, sem_seg_pred):
        rgb = cv2.cvtColor(self.event.frame.copy(), cv2.COLOR_RGB2BGR)#shape (h, w, 3)
        rgb_image = torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).half() / 255

        use_model_0 = abs(self.camera_horizon % 360) <= 5

        if use_model_0:
            _, pred_depth = self.depth_pred_model_0.predict(rgb_image.to(device=self.depth_gpu).float())
            include_mask_prop = self.args.valts_trustworthy_obj_prop0
        else:
            _, pred_depth = self.depth_pred_model.predict(rgb_image.to(device=self.depth_gpu).float())
            include_mask_prop = self.args.valts_trustworthy_obj_prop

        depth = pred_depth.get_trustworthy_depth(
            max_conf_int_width_prop=self.args.valts_trustworthy_prop, include_mask=sem_seg_pred,
            include_mask_prop=include_mask_prop) #default is 1.0
        depth = depth.squeeze().detach().cpu().numpy()

        self.learned_depth_frame = pred_depth.mle().detach().cpu().numpy()[0, 0, :, :]
        # self.learned_depth_frame = pred_depth.expectation().detach().cpu().numpy()[0, 0, :, :]

        if self.args.save_pictures:
            depth_imgname = os.path.join(self.picture_folder_name, '%s', "depth_" + str(self.steps_taken) + ".png")
            depth_pre_mask = (pred_depth.mle()).detach().cpu().numpy()[0, 0, :, :]
            cv2.imwrite(depth_imgname % "depth", depth_pre_mask * 100)
        del pred_depth

        depth = np.expand_dims(depth, 2)
        return depth

    def _preprocess_obs(self, obs):
        # make semantic segmentation and depth predictions

        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]

        sem_seg_pred = self.seg.get_sem_pred(rgb.astype(np.uint8)) #(300, 300, num_cat)

        if self.args.use_learned_depth:
            include_mask = np.sum(sem_seg_pred, axis=2).astype(bool).astype(float)
            include_mask = np.expand_dims(np.expand_dims(include_mask, 0), 0)
            include_mask = torch.tensor(include_mask).to(self.depth_gpu)

            depth = self.depth_pred_later(include_mask)
        else:
            depth = obs[:, :, 3:4]
            if self.args.save_pictures:
                depth_imgname = os.path.join(self.picture_folder_name, '%s', "depth_" + str(self.steps_taken) + ".png")
                cv2.imwrite(depth_imgname % "depth", depth * 100)

        rgb = np.asarray(self.res(rgb.astype(np.uint8)))
        depth = self._preprocess_depth(depth)

        if self.args.save_pictures:
            depth_imgname = os.path.join(self.picture_folder_name, '%s', "depth_" + str(self.steps_taken) + ".png")
            cv2.imwrite(depth_imgname % "depth_thresholded", depth)

        ds = args.env_frame_width // args.frame_width # Downscaling factor
        if ds != 1:
            depth = depth[ds//2::ds, ds//2::ds]
            sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis = 2).transpose(2, 0, 1)

        return state, sem_seg_pred

    def _preprocess_depth(self, depth):
        depth = depth[:, :, 0] * 100

        if self.picked_up:
            mask_err_below = depth < 50
            depth[np.logical_or(self.picked_up_mask, mask_err_below)] = 10000.0

        return depth

    def colorImage(self, sem_map, color_palette):
        semantic_img = Image.new("P", (sem_map.shape[1],
                                       sem_map.shape[0]))

        semantic_img.putpalette(color_palette)
        #semantic_img.putdata((sem_map.flatten() % 39).astype(np.uint8))
        semantic_img.putdata((sem_map.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        semantic_img = np.asarray(semantic_img)
        semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_RGBA2BGR)

        return semantic_img

    def _visualize(self, inputs):
        args = self.args

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                inputs['pose_pred']

        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        if self.steps <=1:
            goal = inputs['goal']
        else:
            goal = self.goal_visualize
        sem_map = inputs['sem_map_pred'].copy()

        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        sem_map += 5

        if self.args.ground_truth_segmentation:
            no_cat_mask = sem_map == 5 + args.num_sem_categories -1
        else:
            no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        map_mask = np.logical_or(map_mask, self.collision_map==1)
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        curr_mask = np.zeros(vis_mask.shape)
        selem = skimage.morphology.disk(2)
        curr_mask[start[0], start[1]] = 1
        curr_mask = 1 - skimage.morphology.binary_dilation(
            curr_mask, selem) != True
        curr_mask = curr_mask ==1
        sem_map[curr_mask] = 3

        if goal is not None:
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1
            sem_map[goal_mask] = 4

        #self.print_log(sem_map.shape, sem_map.min(), sem_map.max())
        #self.print_log(vis_mask.shape)
        #sem_map = self.compress_sem_map(sem_map)

        #color_palette = d3_40_colors_rgb.flatten()
        color_palette2 = [1.0, 1.0, 1.0,
                0.6, 0.6, 0.6,
                0.95, 0.95, 0.95,
                0.96, 0.36, 0.26,
                0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
                0.9400000000000001, 0.7818, 0.66,
                0.9400000000000001, 0.8868, 0.66,
                0.8882000000000001, 0.9400000000000001, 0.66,
                0.7832000000000001, 0.9400000000000001, 0.66,
                0.6782000000000001, 0.9400000000000001, 0.66,
                0.66, 0.9400000000000001, 0.7468000000000001,
                0.66, 0.9400000000000001, 0.9018000000000001,
                0.66, 0.9232, 0.9400000000000001,
                0.66, 0.8182, 0.9400000000000001,
                0.66, 0.7132, 0.9400000000000001,
                0.7117999999999999, 0.66, 0.9400000000000001,
                0.8168, 0.66, 0.9400000000000001,
                0.9218, 0.66, 0.9400000000000001,
                0.9400000000000001, 0.66, 0.9031999999999998,
                0.9400000000000001, 0.66, 0.748199999999999]
        
        color_palette2 += self.flattened.tolist()
        
        color_palette2 = [int(x*255.) for x in color_palette2]
        color_palette = color_palette2

        semantic_img = self.colorImage(sem_map, color_palette)

        arrow_sz = 20
        h, w = semantic_img.shape[:2]
        pt1 = (w - arrow_sz * 2, h - arrow_sz * 2)
        arry, arrx = CH._which_direction(start_o)
        pt2 = (pt1[0] + arrx * arrow_sz, pt1[1] + arry * arrow_sz)
        cv2.arrowedLine(
            semantic_img, pt1, pt2, color=(0, 0, 0, 255), thickness=1,
            line_type=cv2.LINE_AA, tipLength=0.2)

        if self.args.visualize:
            cv2.imshow("Sem Map", semantic_img)
            cv2.waitKey(1)

        if self.args.save_pictures:
            cv2.imwrite(self.picture_folder_name + "Sem_Map/"+ "Sem_Map_" + str(self.steps_taken) + ".png", semantic_img)

    def evaluate(self):
        goal_satisfied = self.get_goal_satisfied()
        if goal_satisfied:
            success = True
        else:
            success = False
            
        pcs = self.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(self.traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(self.steps_taken))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(self.steps_taken))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight
        
        goal_instr = self.traj_data['turk_annotations']['anns'][self.r_idx]['task_desc']
        sliced = get_arguments(self.traj_data)[-1]
        
        # log success/fails
        log_entry = {'trial': self.traj_data['task_id'],
                     #'scene_num': self.traj_data['scene']['scene_num'],
                     'type': self.traj_data['task_type'],
                     'repeat_idx': int(self.r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'sliced':sliced,
                     'episode_no':  self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank,
                     'steps_taken': self.steps_taken}
                     #'reward': float(reward)}
        
        return log_entry, success
