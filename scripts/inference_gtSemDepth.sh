FROM_IDX=$1
TO_IDX=$2
DISPLAY=$3
SPLIT=$4
NAME=$5
SEM_POLICTY_TYPE=$6
MLM_FNAME=$7
MLM_OPTIONS=$8
SEED=$9
GRID_SZ=${10}
TEMPERATURE=${11}
LANG_GRANULARITY=${12}
CENTERING=${13}
TARGET_OFFSET=${14}
OBSTACLE_SELEM=${15}

python main.py \
-n1 \
--max_episode_length 1000 \
--num_local_steps 25 \
--num_processes 2 \
--eval_split ${SPLIT} \
--from_idx ${FROM_IDX} \
--to_idx ${TO_IDX} \
--max_fails 10 \
--debug_local \
--set_dn ${NAME} \
-v 0 \
--which_gpu 0 \
--x_display ${DISPLAY} \
--sem_policy_type ${SEM_POLICTY_TYPE} \
--mlm_fname ${MLM_FNAME} \
--mlm_options ${MLM_OPTIONS} \
--seed ${SEED} \
--splits alfred_data_small/splits/oct21.json \
--grid_sz ${GRID_SZ} \
--mlm_temperature ${TEMPERATURE} \
--approx_last_action_success \
--language_granularity ${LANG_GRANULARITY} \
--centering_strategy ${CENTERING} \
--target_offset_interaction ${TARGET_OFFSET} \
--obstacle_selem ${OBSTACLE_SELEM}
