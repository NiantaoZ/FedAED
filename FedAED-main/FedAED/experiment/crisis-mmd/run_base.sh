for alpha in 1.0; do 
    for fed_alg in FAED; do
        taskset -c 1-60 python3 train.py --alpha $alpha --hid_size 128 --sample_rate 0.1 --learning_rate 0.05  --global_learning_rate 0.004 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg --mu 0.01
	#taskset -c 1-60 python3 train.py --alpha $alpha --hid_size 128 --sample_rate 0.1 --learning_rate 0.05  --global_learning_rate 0.004 --num_epochs 300 --fed_alg $fed_alg --mu 0.01
    done
done
