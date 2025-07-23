# bash scripts/script_train_single.sh vrvq/vrvq_a2 0
# bash scripts/script_train_multi.sh vrvq/vrvq_a2 0,1,2,3,4,5,6,7
nohup bash scripts/script_train_multi.sh vrvq/vrvq_a2 0,1,2,3,4,5,6,7 > training_vrvq_a2.log 2>&1 &