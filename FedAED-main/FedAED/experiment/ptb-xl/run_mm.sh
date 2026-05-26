for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
    for fed_alg in pFedMe peravg FedBN; do
        taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --global_learning_rate 0.01 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg --mu 0.01 --en_missing_modality --missing_modailty_rate $missing_rate
	#taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --global_learning_rate 0.01 --num_epochs 300 --fed_alg $fed_alg --mu 0.01 --en_missing_modality --missing_modailty_rate $missing_rate
    done
done
