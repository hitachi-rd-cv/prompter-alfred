from ast import parse
from cgitb import small
import os
import re
import pickle
import numpy as np
import pandas as pd

from functools import reduce


def getSuccessPathLen(result_fnames, ep2success):
    results = list()
    for fname in result_fnames:
        if os.path.exists(fname):
            result = pickle.load(open(fname, 'rb'))
            results += result

    if len(results) == 0:
        return None

    pathlens = list()
    for result in results:
        key = list(result.keys())[0]
        epnum, _ = key
        assert epnum in ep2success

        if ep2success[epnum]:
            pathlens.append(len(result[key]))

    return pathlens


def aggrDF(result_fnames, kword):
    results = list()
    for fname in result_fnames:
        if os.path.exists(fname):
            result = pickle.load(open(fname, 'rb'))
            results += result

    if len(results) == 0:
        return None

    df = pd.DataFrame(results)
    dups = df[kword].duplicated()
    df = df.drop(df[dups].index)
    df = df.sort_values(kword)

    return df


def parseLog(result_fnames):
    log_lines = list()
    for fname in result_fnames:
        if os.path.exists(fname):
            with open(fname, "r") as f:
                log_lines += f.readlines()

    details = dict()
    for line in log_lines:
        line = line.strip()
        if len(line) == 0:
            continue

        if "episode # is" in line:
            curr_episode = int(re.findall("[0-9]+", line)[0])
            details[curr_episode] = {
                "details": dict(), "actions": "", "cat2idx": "", "cat2idx_str": "", "success": False,
                "errors": list()}
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
                step_detail = line[len(step_inst):]

                if "err is" in line:
                    err_header = re.findall("step: [0-9]+ err is ", step_detail)[0]
                    details[curr_episode]["errors"].append(step_detail[len(err_header):])

                if step not in details[curr_episode]["details"]:
                    details[curr_episode]["details"][step] = list()
                details[curr_episode]["details"][step].append(step_detail)

    return details


def cleanError(message, ignore_words):
    if "is blocking Agent 0 from moving 0" in message:
        return "blocked by another object"

    cleaned_message = list()
    words = message.split(' ')
    for word in words:
        bad_word = False
        for ignore in ignore_words:
            if ignore in word:
                bad_word = True
                break
        if len(re.findall("FP[0-9][0-9][0-9]", word)) > 0:
            bad_word = True
        if not bad_word:
            cleaned_message.append(word)

    return ' '.join(cleaned_message)


def errorAnalysis(log_data, ignore_words):
    episodes = sorted(log_data.keys())

    error_counts = dict()
    error_counts_weighted = dict()
    for episode in episodes:
        errors = [cleanError(err, ignore_words) for err in log_data[episode]["errors"]]

        for error in errors:
            if error not in error_counts:
                error_counts[error] = 0
                error_counts_weighted[error] = 0
            error_counts[error] += 1

        error_set = set(errors)
        for error in error_set:
            error_counts_weighted[error] += 1

    return error_counts, error_counts_weighted


if __name__ == '__main__':
    results_dname = "results"
    use_log = False
    suffixes = {"even": ["testrun"]}

    split = "valid_unseen"

    suffix_list = list()
    dfs, dfs_weighted, dfs_weighted_fail, path_lens = dict(), dict(), dict(), dict()
    log_info = dict()
    for indx_range, sfxs in suffixes.items():
        for suffix in sfxs:
            leaderboard_tmpl = None
            if indx_range == "even":
                result_tmpl = [
                    f'{split}_%s_0_to_821_{suffix}.p'
                ]
            name = f"{indx_range}_{suffix}"
            suffix_list.append(name)
            dfs[name] = aggrDF(
                [os.path.join(results_dname, 'analyze_recs', rt %
                              "anaylsis_recs_from") for rt in result_tmpl],
                "number_of_this_episode")
            dfs_weighted[name] = aggrDF(
                [os.path.join(results_dname, 'successes', rt %
                              "successes_from") for rt in result_tmpl],
                "episode_no")
            dfs_weighted_fail[name] = aggrDF(
                [os.path.join(results_dname, 'fails', rt %
                              "failures_from") for rt in result_tmpl],
                "episode_no")
            if use_log:
                log_info[name] = parseLog(
                    [os.path.join(results_dname, 'logs', ("log_" + rt.replace(".p", ".txt")) % "from")
                    for rt in result_tmpl])
            if (leaderboard_tmpl is not None) and (dfs_weighted[name] is not None):
                df = dfs[name]
                ep2success = {epnum: succ for epnum, succ in
                              zip(df["number_of_this_episode"], df["success"])}
                path_lens[name] = getSuccessPathLen(
                    [os.path.join(results_dname, 'leaderboard', rt)
                                  for rt in leaderboard_tmpl],
                    ep2success)
            else:
                path_lens[name] = None

    all_obj = ['ArmChair', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 'Microwave', 'Ottoman', 'Safe', 'Shelf', 'SideTable', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toilet', 'AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
    ignore_words = all_obj

    df_mod = dict()
    df_result = list()
    error_result = list()
    error_weighted_result = list()
    for suffix in suffix_list:
        df, df_weighted, df_weighted_fail, path_len = dfs[suffix], dfs_weighted[suffix], dfs_weighted_fail[suffix], path_lens[suffix]
        num_samples = len(df)

        # success rate
        sr = df["success"].mean()
        print(f"{suffix}")
        print(f"\tSR: {df['success'].sum()}/{len(df)} = {sr}")

        if df_weighted is not None:
            srw = df_weighted["success_spl"].sum() / num_samples
            pathlengths = df_weighted["steps_taken"]

            print(f"\tWeighted SR: {df_weighted['success_spl'].sum()}/{num_samples} = {srw}")
            print(f"\tMean/median path length: {pathlengths.mean()}/{pathlengths.median()}")

            gc = (df_weighted["goal_condition_success"].sum() + df_weighted_fail["goal_condition_success"].sum()) / num_samples
            gcw = (df_weighted["goal_condition_spl"].sum() + df_weighted_fail["goal_condition_spl"].sum()) / num_samples

            run_df = dict()
            run_df["name"] = suffix
            run_df["SR"] = sr
            run_df["GC"] = gc
            run_df["SRPLW"] = srw
            run_df["GCPLW"] = gcw
            run_df["pathlengths_mean"] = pathlengths.mean()
            run_df["pathlengths_median"] = pathlengths.median()
            df_result.append(run_df)

            # error analysis from log files
            if suffix in log_info:
                log_data = log_info[suffix]
                error_counts, error_counts_weighted = errorAnalysis(log_data, ignore_words)
                error_counts["name"] = suffix
                error_counts_weighted["name"] = suffix
                error_result.append(error_counts)
                error_weighted_result.append(error_counts_weighted)

        if path_len is not None:
            path_len = np.asarray(path_len)
            print(f"\tnum_success: {len(path_len)}")
            print(f"\tPath len sum: {path_len.sum()}")
            print(f"\tPath len avg: {path_len.mean()}")

        # rename the success column name and drop all columns except for
        # the success and number_of_this_episode columns
        df_mod[suffix] = df.rename(columns={"success": suffix})[
            ["number_of_this_episode", suffix]]

    redux = reduce(lambda left, right: pd.merge(left, right, on=["number_of_this_episode"], how="outer"),
                   df_mod.values())
    redux.to_csv(os.path.join(
        results_dname, "rslt_successes.csv"), index=False)

    df_result = pd.DataFrame(df_result)
    df_result.to_csv(os.path.join(results_dname, "rslt.csv"), index=False)

    if len(error_result) > 0:
        df_error = pd.DataFrame(error_result)
        df_error.to_csv(os.path.join(results_dname, "error.csv"), index=False)

        df_errorW = pd.DataFrame(error_weighted_result)
        df_errorW.to_csv(os.path.join(results_dname, "error_per_episode.csv"), index=False)
