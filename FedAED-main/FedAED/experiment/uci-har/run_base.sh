for alpha in 0.1 0.5 5.0; do
   for fed_alg in APFL FedMKD; do
      taskset -c 1-30 python3 train.py --alpha $alpha --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 300 --fed_alg $fed_alg --mu 0.01 --hid_size 128
      #taskset -c 1-30 python3 train.py --alpha $alpha --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 300 --fed_alg $fed_alg --mu 0.01 --en_att --att_name fuse_base --hid_size 128
   done
done
#fed_avg fed_opt fed_prox

