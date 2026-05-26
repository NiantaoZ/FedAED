for fed_alg in FAED; do
    #taskset -c 1-30 python3 train.py --sample_rate 0.1 --hid_size 128 --learning_rate 0.01 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg  --global_learning_rate 0.002 --mu 0.1 
    taskset -c 1-30 python3 train.py --sample_rate 0.1 --hid_size 128 --learning_rate 0.01 --num_epochs 300 --fed_alg $fed_alg --global_learning_rate 0.002 --mu 0.1
done
