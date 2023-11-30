export PDSH_SSH_ARGS_APPEND=''
deepspeed --num_nodes=2 \
--num_gpus=1 \
--hostfile hostfile.txt \
--launcher pdsh \
--master_addr 10.254.21.43 \
--ssh_port 2222 textsql_train.py \
--model_id 'google/flan-t5-large' \
--dataset_path my_dataset \
--outputdir trainoutput-wikisql \
--epochs 5 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--lr 1e-4 \
--deepspeed dsconfig/zero2.json