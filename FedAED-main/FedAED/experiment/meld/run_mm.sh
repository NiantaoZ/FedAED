for missing_rate in 0.3; do
    for fed_alg in APFL FedMKD; do
        #taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.002 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate 
	taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.002 --num_epochs 300 --fed_alg $fed_alg --en_label_nosiy --label_nosiy_level $missing_rate
    done
done
