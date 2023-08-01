import cv2
import math

import numpy as np
import utils.control_helper as CH

from numba import jit

from scipy import sparse
from scipy.sparse.csgraph import shortest_path


def pose2indx(y, x, o, map_shape):
    o = o % 360
    assert o in [0, 90, 180, 270], f"orientation should be one of [0, 90, 180, 270], but is {o}"

    num_coords = map_shape[0] * map_shape[1]
    return int(y * map_shape[1] + x + int(o / 90) * num_coords)


def indx2pose(indx, map_shape):
    num_coords = map_shape[0] * map_shape[1]
    coord_indx = indx % num_coords
    return (math.floor(coord_indx / map_shape[1]), coord_indx % map_shape[1], math.floor(indx / num_coords) * 90)


def convert2connectivityPose(pose, step_size):
    pose = [pose[0], pose[1], 0] if len(pose) == 2 else pose
    return (round(pose[0] / step_size), round(pose[1] / step_size), pose[2] % 360)


def convert2traversibleCoord(pose, step_size):
    return [0, 0] if pose is None else [nc * step_size for nc in pose[:2]]


def inBounds(coord, height, width):
    return (max(0, min(height-1, coord[0])), max(0, min(width-1, coord[1])))


def isInBounds(coord, height, width):
    return (coord[0] >= 0) and (coord[1] >= 0) and (coord[0] < height) and (coord[1] < width)


def pose2coord(pose):
    return None if pose is None else [pose[0], pose[1]]


@jit(nopython=True)
def map2connectivity(traversible, step_size):
    h, w = traversible.shape[:2]
    h_sz, w_sz = h // step_size + 1, w // step_size + 1
    num_nodes = h_sz * w_sz * 4

    connectivity = np.eye(num_nodes, dtype=np.bool_)
    for y in range(h_sz):
        for x in range(w_sz):
            for o in range(0, 360, 90):
                c_indx = int(y * w_sz + x + int((o % 360) / 90) * h_sz * w_sz)

                # RotateLeft and RotateRight
                c_indx_left = int(y * w_sz + x + int(((o + 90) % 360) / 90) * h_sz * w_sz)
                c_indx_right = int(y * w_sz + x + int(((o - 90) % 360) / 90) * h_sz * w_sz)
                connectivity[c_indx, c_indx_left] = 1
                connectivity[c_indx, c_indx_right] = 1

                # MoveAhead
                if o == 0:
                    dy, dx = 0, 1
                elif o == 90:
                    dy, dx = 1, 0
                elif o == 180:
                    dy, dx = 0, -1
                elif o == 270:
                    dy, dx = -1, 0
                neighbor_y, neighbor_x = max(0, min(h_sz-1, y+dy)), max(0, min(w_sz-1, x+dx))
                lower_x, upper_x = min(neighbor_x, x), max(neighbor_x, x)
                lower_y, upper_y = min(neighbor_y, y), max(neighbor_y, y)
                roi = traversible[lower_y * step_size:upper_y * step_size + 1,
                                  lower_x * step_size:upper_x * step_size + 1]
                is_connected = (roi.sum() == (step_size + 1))

                t_indx = int(neighbor_y * w_sz + neighbor_x + int((o % 360) / 90) * h_sz * w_sz)
                connectivity[c_indx, t_indx] = is_connected

    return connectivity


def map2connectivityNoNumba(traversible, step_size):
    # non-numba version of map2connectivity
    # should be more readable

    h, w = traversible.shape[:2]
    h_sz, w_sz = h // step_size + 1, w // step_size + 1
    num_nodes = h_sz * w_sz * 4

    connectivity = np.eye(num_nodes, dtype=bool)
    for y in range(h_sz):
        for x in range(w_sz):
            for o in range(0, 360, 90):
                c_indx = pose2indx(y, x, o, (h_sz, w_sz))

                # RotateLeft and RotateRight
                connectivity[c_indx, pose2indx(y, x, (o + 90) % 360, (h_sz, w_sz))] = 1
                connectivity[c_indx, pose2indx(y, x, (o - 90) % 360, (h_sz, w_sz))] = 1

                # MoveAhead
                dy, dx = CH._which_direction(o)
                neighbor_y, neighbor_x = inBounds((y + dy, x + dx), h_sz, w_sz)
                lower_x, upper_x = min(neighbor_x, x), max(neighbor_x, x)
                lower_y, upper_y = min(neighbor_y, y), max(neighbor_y, y)
                roi = traversible[lower_y * step_size:upper_y * step_size + 1,
                                  lower_x * step_size:upper_x * step_size + 1]
                is_connected = (roi.sum() == (step_size + 1))

                t_indx = pose2indx(neighbor_y, neighbor_x, o, (h_sz, w_sz))
                connectivity[c_indx, t_indx] = is_connected

    return connectivity


def getCandCoords(pt, distance, map_shape, kernel_shape=cv2.MORPH_CROSS):
    candidate_map = np.zeros(map_shape, dtype=np.uint8)
    candidate_map[pt[0], pt[1]] = 1

    if distance != 0:
        kernel_sz = 2 * distance + 1
        prev_kernel = cv2.getStructuringElement(kernel_shape, (kernel_sz - 2, kernel_sz - 2))
        kernel = cv2.getStructuringElement(kernel_shape, (kernel_sz, kernel_sz))
        candidate_map = cv2.dilate(candidate_map, kernel) - cv2.dilate(candidate_map, prev_kernel)

    return np.where(candidate_map)


def findReachablePose(goal, dist_thresh, step_size, map_shape, dist_matrix, search_shape=cv2.MORPH_CROSS):
    for displacement in range(0, 30 // step_size):
        distances, cand_indices = list(), list()
        cand_ys, cand_xs = getCandCoords(goal, displacement, map_shape, search_shape)
        for cand_y, cand_x in zip(cand_ys, cand_xs):
            for orientation in range(0, 360, 90):
                cand_indx = pose2indx(cand_y, cand_x, orientation, map_shape)
                distances.append(dist_matrix[cand_indx])
                cand_indices.append(cand_indx)

        min_indx = np.argmin(distances)
        min_dist = distances[min_indx]
        if (min_dist != np.inf) and ((dist_thresh is None ) or (min_dist < dist_thresh)):
            return indx2pose(cand_indices[min_indx], map_shape)

    return None


def isReachable(end_indx, dist_matrix):
    return dist_matrix[end_indx] != np.inf


def getReachableGoalPose(goal, step_size, map_shape, dist_matrix):
    dist_thresh = 200 // step_size
    # dist_thresh = 125 // step_size
    goal_coordinate = findReachablePose(
        goal, dist_thresh, step_size, map_shape, dist_matrix,
        search_shape=cv2.MORPH_CROSS)

    # if searching in cross shape failed, try circular shape
    if goal_coordinate is None:
        goal_coordinate = findReachablePose(
            goal, dist_thresh, step_size, map_shape, dist_matrix,
            search_shape=cv2.MORPH_ELLIPSE)

    # if candidate search failed, ask for a new goal
    if goal_coordinate is None:
        return None

    return goal_coordinate


def nextStep(predecessors, start_indx, goal_indx, map_shape, step_size):
    path = []
    i = goal_indx
    while i != start_indx:
        path.append(i)
        i = predecessors[i]
    path.append(i)
    path = path[::-1]

    next_indx = path[1]
    next_pose = indx2pose(next_indx, map_shape)
    start_pose = indx2pose(start_indx, map_shape)

    angle_diff = (next_pose[2] - start_pose[2]) % 360
    if angle_diff == 0:
        return "MoveAhead_25"
    if angle_diff == 90:
        return "RotateLeft_90"
    if angle_diff == 270:
        return "RotateRight_90"


def shiftGoal(old_goal, new_goal, traversible, target_offset, step_size, measure_offset_from_edge):
    # shift the selected coordinate by target_offset, so that the agent
    # is specific distance away from the target
    if (new_goal is None) or np.array_equal(old_goal, new_goal):
        return (new_goal[0], new_goal[1], 0)

    old_goal = np.asarray(convert2traversibleCoord(old_goal, step_size))
    new_goal = np.asarray(convert2traversibleCoord(new_goal, step_size))
    shift_vector = new_goal - old_goal

    if not measure_offset_from_edge:
        dist = np.sqrt(shift_vector[0] ** 2 + shift_vector[1] ** 2)
        target_offset = max(0, round(target_offset - dist))

    h, w = traversible.shape[:2]
    angle = np.round(np.rad2deg(np.arctan2(shift_vector[0], shift_vector[1])) / 90) * 90
    unit_vec = np.asarray(CH._which_direction(angle))
    for shift in range(0, target_offset + 1):
        shifted_goal = new_goal + unit_vec * shift
        if (not isInBounds(shifted_goal, h, w)) or (not traversible[shifted_goal[0], shifted_goal[1]]):
            break

    shift = max(0, np.floor((shift - 1) / step_size) * step_size)
    shifted_goal = (new_goal + unit_vec * shift).astype(int)
    return convert2connectivityPose(shifted_goal, step_size)


def planNextMove(traversible, step_size, start, goal, target_offset, measure_offset_from_edge):
    # calculate the connectivity graph
    connectivity = map2connectivity(traversible, step_size)

    # same as the above code, but more readable
    # connectivity2 = map2connectivityNoNumba(traversible, step_size)
    # assert np.array_equal(connectivity, connectivity2), "sfdsafaa"

    connectivity = sparse.csr_matrix(connectivity)

    start_pose = convert2connectivityPose(start, step_size)
    goal_pose = convert2connectivityPose(goal, step_size)

    map_shape = [(dim // step_size + 1) for dim in traversible.shape[:2]]
    start_indx = pose2indx(start_pose[0], start_pose[1], start_pose[2], map_shape)

    dist_matrix, predecessors = shortest_path(
        connectivity, return_predecessors=True, indices=start_indx)

    goal_fixed = (len(goal) == 3)
    next_goal = False
    if goal_fixed:
        goal_free_and_not_shifted = False
        arrived_at_goal = (start_pose == goal_pose)

        if arrived_at_goal:
            return "LookUp_0", start, True, next_goal, False
        else:
            goal_indx = pose2indx(goal_pose[0], goal_pose[1], goal_pose[2], map_shape)
            new_goal = convert2traversibleCoord(goal_pose, step_size)

    else:
        # search for a goal coordinate that is reachable
        reachable_goal_pose = getReachableGoalPose(
            pose2coord(goal_pose), step_size, map_shape, dist_matrix)
        # self.print_log(f"goal shifted from {ori_goal_coord[1]},{ori_goal_coord[0]} to {goal_coordinate[1]},{goal_coordinate[0]} to avoid obstacles")

        if reachable_goal_pose is None:
            return None, start, True, True, False

        # shift the goal, so that the agent is not too close to the target object
        new_goal_pose = shiftGoal(
            pose2coord(goal_pose), pose2coord(reachable_goal_pose), traversible,
            target_offset, step_size, measure_offset_from_edge)

        goal_not_shifted = (pose2coord(goal_pose) == pose2coord(new_goal_pose))
        goal_free = traversible[goal[0], goal[1]]
        goal_free_and_not_shifted = goal_free and goal_not_shifted
        arrived_at_goal = ((pose2coord(start_pose) == pose2coord(new_goal_pose))) or (new_goal_pose is None)

        if arrived_at_goal:
            return "LookUp_0", start, True, next_goal, goal_free_and_not_shifted
        else:
            # figure out what the goal pose is, after shifting happens
            # find the closest pose to current location, if the goal does not specify orientation
            goal_indices = [pose2indx(new_goal_pose[0], new_goal_pose[1], orientation, map_shape)
                            for orientation in range(0, 360, 90)]
            goal_indx = goal_indices[np.argmin(dist_matrix[goal_indices])]
            new_goal = convert2traversibleCoord(new_goal_pose, step_size)

    # plan the next movement
    nextAction = nextStep(predecessors, start_indx, goal_indx, map_shape, step_size)
    stop = False

    return nextAction, new_goal, stop, next_goal, goal_free_and_not_shifted
