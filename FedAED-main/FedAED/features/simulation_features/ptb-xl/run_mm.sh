for mm_rate in 0.1 0.2 0.3 0.4 0.5; do
    #taskset -c 1-30 python3 simulation_feature.py --en_missing_modality --missing_modailty_rate $mm_rate
    #taskset -c 1-30 python3 simulation_feature.py --en_missing_label --missing_label_rate $mm_rate
    taskset -c 1-30 python3 simulation_feature.py --en_label_nosiy --label_nosiy_level $mm_rate
done
