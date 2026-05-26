for fed_alg in APFL FedMKD; do
    taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg  --global_learning_rate 0.01 --mu 0.01
    #taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --num_epochs 300 --fed_alg $fed_alg  --global_learning_rate 0.01 --mu 0.01
done
