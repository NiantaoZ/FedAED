for missing_rate in 0.4; do
    for fed_alg in fed_opt; do
       #taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg --en_label_nosiy --label_nosiy_level $missing_rate --en_missing_label --missing_label_rate $missing_rate --en_missing_modality --missing_modailty_rate $missing_rate
       taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 140 --fed_alg $fed_alg --en_missing_label --missing_label_rate $missing_rate
    done
done
