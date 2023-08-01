python end_to_end_outputs.py -sp val_unseen -m models/best_models_noappended -o instruction2_params_val_unseen_noappended --no_appended
python end_to_end_outputs.py -sp val_seen -m models/best_models_noappended -o instruction2_params_val_seen_noappended --no_appended
python end_to_end_outputs.py -sp tests_unseen -m models/best_models_noappended -o instruction2_params_test_unseen_noappended --no_appended
python end_to_end_outputs.py -sp tests_seen -m models/best_models_noappended -o instruction2_params_test_seen_noappended --no_appended

python end_to_end_outputs.py -sp val_unseen -m models/best_models -o instruction2_params_val_unseen_appended
python end_to_end_outputs.py -sp val_seen -m models/best_models -o instruction2_params_val_seen_appended
python end_to_end_outputs.py -sp tests_unseen -m models/best_models -o instruction2_params_test_unseen_appended
python end_to_end_outputs.py -sp tests_seen -m models/best_models -o instruction2_params_test_seen_appended
