for max_number_models in [5, 10, 20, 40]:
    for max_number_folds in [3, 5, 10]:
        for max_num_repetitions in [1, 3, 5]:
            for stacking_levels in [1, 2]:
                for cv_type in ['cv', 'partial-cv']:
                    for allowed_folds_to_use in [0.1, 0.5, 1.0]:
                        cmd = "python manager_grid.py --framework autosklearn -b master_thesis --partition bosch_cpu-cascadelake --total_fold 5 --cores 1 --memory 8G --runtime 3600  --max_number_models {} --max_number_folds {} --max_num_repetitions {} --stacking_levels {} --cv_type {} --seed 42 --allowed_folds_to_use {} --task Australian --run_mode single".format(
                            max_number_models,
                            max_number_folds,
                            max_num_repetitions,
                            stacking_levels,
                            cv_type,
                            allowed_folds_to_use,
                        )
                        if cv_type != 'cv' or allowed_folds_to_use != 1.0:
                            print(f"#{cmd}")
                        else:
                            print(f"{cmd}")
