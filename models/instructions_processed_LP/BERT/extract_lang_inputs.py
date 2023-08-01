import os
import json
import pickle
import string


exclude = set(string.punctuation)
def cleanInstruction(instruction):
    instruction = instruction.lower()
    instruction = ''.join(ch for ch in instruction if ch not in exclude)
    return instruction


index_jsonfname = "../../../alfred_data_small/splits/oct21.json"
trajectory_dname = "../../../alfred_data_all/json_2.1.0"
with open(index_jsonfname, 'rb') as f:
    index_json = json.load(f)

info_dict = dict()
for split_name, split_contents in index_json.items():
    split_name = split_name.replace("valid", "val")
    print(split_name)
    if split_name == "train":
        continue

    split_dict = {"x": list(), "x_low": list()}
    for index_info in split_contents:
        traj_jsonname = os.path.join(
            trajectory_dname, index_info['task'], "pp", f"ann_{index_info['repeat_idx']}.json")
        with open(traj_jsonname, 'rb') as f:
            traj_json = json.load(f)

        info = traj_json["turk_annotations"]["anns"][index_info['repeat_idx']]

        high_lang = cleanInstruction(info["task_desc"])
        low_lang = [cleanInstruction(inst) for inst in info["high_descs"]]

        split_dict["x"].append(high_lang)
        split_dict["x_low"].append('[SEP]'.join([high_lang] + low_lang))

    info_dict[split_name] = split_dict

with open("data/input_lang.p", 'wb') as f:
    pickle.dump(info_dict, f)
