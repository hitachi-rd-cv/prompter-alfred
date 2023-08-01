import os
import re
import ast
import cv2
import copy
import json

import numpy as np

from textwrap import wrap
from viz_utils import ImgElem, TxtElem, Vid


class Step:
    def __init__(self, num):
        self.num = num
        self.info = list()

    def addInfo(self, info):
        self.info.append(info)


def parseLog(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    details = dict()
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        if "episode # is" in line:
            curr_episode = int(re.findall("[0-9]+", line)[0])
            details[curr_episode] = {
                "details": dict(), "actions": "", "cat2idx": "", "cat2idx_str": "", "success": False}
        elif "list of actions is" in line:
            details[curr_episode]["actions"] = line[line.find('['):]
        elif "total cat2idx" in line:
            details[curr_episode]["cat2idx"] = [sss[1:-1]
                                                for sss in re.findall("'[a-zA-Z]+'", line[line.find('{'):])]
            details[curr_episode]["cat2idx_str"] = line[line.find('{'):]
        elif ("success is True" in line) or ("This episode is probably Success!" in line):
            details[curr_episode]["success"] = True
        else:
            step_insts = re.findall("step #:  [0-9]+ , ", line)
            if len(step_insts) > 0:
                step_inst = step_insts[0]
                step = int(re.findall("[0-9]+", step_inst)[0])
                if step not in details[curr_episode]["details"]:
                    details[curr_episode]["details"][step] = list()
                details[curr_episode]["details"][step].append(
                    line[len(step_inst):])

    return details


def parseSplitInfo(split_name, split_info_fname):
    with open(split_info_fname) as f:
        split_info = json.load(f)[split_name]

    return [v["task"] for v in split_info]


def filterRunsByLength(lbound, ubound, log_details):
    rlength = [max(ldet["details"].keys()) for _, ldet in log_details.items()]
    run_indx = list(log_details.keys())

    return [indx for indx, rl in zip(run_indx, rlength) if ((rl > lbound) and (rl < ubound))]


def main():
    split_name = "valid_unseen"
    run_range = "0_to_821"
    run_name = "testrun"
    runs = [0, 20]

    split_info_fname = "../alfred_data_small/splits/oct21.json"

    log_details = parseLog(
        f"../results/logs/log_{split_name}_from_{run_range}_{run_name}.txt")

    left_x = 10
    right_x = 1250
    imgs_y = 40
    imgs_y2 = imgs_y + 350
    imgs_y3 = imgs_y2 + 350

    base_imgd = f"../pictures/{split_name}/{run_name}"
    trial_ids = parseSplitInfo(split_name, split_info_fname=split_info_fname)
    datad = f"../alfred_data_all/json_2.1.0/{split_name}"
    for trial_num, trial_id in enumerate(trial_ids):
        if (runs is not None) and (trial_num not in runs):
            continue
        if (trial_num not in log_details):
            continue

        print(trial_num, trial_id)

        # load logs
        info_strs = list()
        with open(os.path.join(datad, trial_id, "traj_data.json"), "rb") as f:
            data = json.load(f)
        info_strs.append(f"Floor plan: {data['scene']['floor_plan']}")
        info_strs.append("Samples of high-level descriptions:")
        info_strs += [f"  {ann['task_desc']}" for ann in data["turk_annotations"]["anns"]]

        low_descs = ["One sample of low-level description:"]
        low_descs += [f"  {ld}" for ld in data["turk_annotations"]
                      ["anns"][0]["high_descs"]]

        log_det = log_details[trial_num]
        goals = ast.literal_eval(log_det['actions'])

        imgd = os.path.join(base_imgd, str(trial_num))

        # set the video elements
        # raw info
        rgbElem = ImgElem(
            dirname=os.path.join(imgd, "rgb"),
            fname_template="rgb_%d.png", pos=(right_x, imgs_y), text="RGB")

        # processed info
        semElem = ImgElem(
            dirname=os.path.join(imgd, "Sem"),
            fname_template="Sem_%d.png", pos=(right_x, imgs_y2), text="semantic segmentation")

        depthElem = ImgElem(
            dirname=os.path.join(imgd, "depth"),
            fname_template="depth_%d.png", pos=(right_x + 400, imgs_y2), text="depth estimation")

        depth_threshElem = ImgElem(
            dirname=os.path.join(imgd, "depth_thresholded"),
            fname_template="depth_%d.png", pos=(right_x + 800, imgs_y2), text="depth estimation w/ threshold")

        # interpreted info
        fmm_distElem = ImgElem(
            dirname=os.path.join(imgd, "fmm_dist"), do_flip=True,
            fname_template="fmm_dist_%d.png", pos=(right_x, imgs_y3), text="obstacles")

        sem_mapElem = ImgElem(
            dirname=os.path.join(imgd, "Sem_Map"), do_flip=True,
            fname_template="Sem_Map_%d.png", pos=(right_x + 400, imgs_y3), text="semantic map")

        sem_map_targetElem = ImgElem(
            dirname=os.path.join(imgd, "Sem_Map_Target"), do_flip=True,
            fname_template="Sem_Map_Target_%d.png", pos=(right_x + 800, imgs_y3), text="observed target location")

        frame_n_elem = TxtElem(pos=(left_x, 0), size=1.5, thickness=2)
        success_elem = TxtElem(contents=f"Successful?: {log_det['success']}", pos=(left_x + 500, 15))
        info_elem = TxtElem(contents=info_strs, pos=(left_x, 60))
        low_desc_elem = TxtElem(contents=low_descs, pos=(left_x, 230), size=0.8, thickness=1)
        goal_elem = TxtElem(pos=(left_x, 390), size=0.7, thickness=1, newline_down=True)
        log_elem = TxtElem(pos=(left_x, 1150), newline_down=False)

        vidname = f'{trial_num}_{trial_id.replace("/", "_")}.avi'
        vid = Vid(vidname=vidname, fps=5, dim=(2600, 1200),
                  elems=[
                    fmm_distElem, rgbElem, semElem, sem_mapElem, sem_mapElem,
                    sem_map_targetElem, depthElem, depth_threshElem,
                    frame_n_elem, success_elem, info_elem, low_desc_elem, log_elem, goal_elem])

        wrap_sz = 80
        curr_goal_num = 0
        num_frames = vid.getMaxFrame()
        for frame_n in range(num_frames):
            frame_n_elem.update(f"Frame #{frame_n}")

            if frame_n in log_det["details"]:
                log_strs = log_det["details"][frame_n]
                log_elem.update(log_strs)

                # update currect goal
                for det in log_strs:
                    if "pointer increased goal name" in det:
                        goal_a, goal_b = ast.literal_eval(
                            det[det.find("'") - 1:])
                        while True:
                            curr_goal_num += 1
                            if len(goals) == curr_goal_num:
                                curr_goal_num -= 1
                                break
                            curr_a, curr_b = goals[curr_goal_num]
                            if goal_a == curr_a and goal_b == curr_b:
                                break
            else:
                log_elem.update(None)

            # logs
            cat2idx = ', '.join(log_det['cat2idx'])
            annotated_goal = copy.deepcopy(goals)
            annotated_goal[curr_goal_num] = f"*{annotated_goal[curr_goal_num]}*"
            action_str = str(annotated_goal)
            log_strs = ["List of actions:"] + \
                wrap(action_str, wrap_sz) + \
                ["", "Categories:"] + wrap(cat2idx, wrap_sz)
            goal_elem.update(log_strs)

            vid.draw(frame_n)

        vid.release()


if __name__ == "__main__":
    main()
