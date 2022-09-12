#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=def-chdesa
save_dir=save_0912
declare -a StringArray=(

#upper_baseline
"python -O toySeg_cons.py seed=123 lr=0.000001 weight_epoch=100 ratio=0.9 save_dir=${save_dir}/symmetry_upper501lr"
"python -O toySeg_cons.py seed=123 lr=0.00001  weight_epoch=100 ratio=0.9 save_dir=${save_dir}/symmetry_upper401lr"
"python -O toySeg_cons.py seed=123 lr=0.0001   weight_epoch=100 ratio=0.9 save_dir=${save_dir}/symmetry_upper301lr"

#lower_baseline
#"python -O toySeg_cons.py seed=123 lr=0.00001 weight_epoch=100 ratio=0.1 save_dir=${save_dir}/symmetry_lower_10"
#"python -O toySeg_cons.py seed=123 lr=0.00001 weight_epoch=100 ratio=0.3 save_dir=${save_dir}/symmetry_lower_30"
#"python -O toySeg_cons.py seed=123 lr=0.00001 weight_epoch=100 ratio=0.5 save_dir=${save_dir}/symmetry_lower_50"

# cons
#"python -O toySeg_cons.py seed=123 train_mode=cons lr=0.00001 weight_epoch=15 ratio=0.3 weights.cons_weight=0.5 weights.weight_cons=0.005 save_dir=${save_dir}/symmetry_cons "

# vat
#"python -O toySeg_cons.py seed=123 train_mode=vat lr=0.00001 weight_epoch=15 ratio=0.3 weights.cons_weight=0.5 weights.weight_adv=0.005 save_dir=${save_dir}/symmetry_vat "

# cat
#"python -O toySeg_cons.py seed=123 train_mode=cat lr=0.00001 weight_epoch=15 ratio=0.3 weights.cons_weight=0.5 weights.weight_adv=0.005 weights.weight_cons=0.005 save_dir=symmetry_ "

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

