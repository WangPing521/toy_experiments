#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=save_0913
declare -a StringArray=(

#upper_baseline
#"python -O toySeg_cons.py seed=123 lr=0.001 weight_epoch=100 ratio=0.9 save_dir=${save_dir}/symmetry_upper201lr"
#"python -O toySeg_cons.py seed=123 lr=0.005 weight_epoch=100 ratio=0.9 save_dir=${save_dir}/symmetry_upper205lr"

#lower_baseline
"python -O toySeg_cons.py seed=123 lr=0.001 weight_epoch=100 ratio=0.1 save_dir=${save_dir}/symmetry_lower_10_201lr"
"python -O toySeg_cons.py seed=123 lr=0.001 weight_epoch=100 ratio=0.3 save_dir=${save_dir}/symmetry_lower_30_201lr"
"python -O toySeg_cons.py seed=123 lr=0.001 weight_epoch=100 ratio=0.5 save_dir=${save_dir}/symmetry_lower_50_201lr"

"python -O toySeg_cons.py seed=123 lr=0.005 weight_epoch=100 ratio=0.1 save_dir=${save_dir}/symmetry_lower_10_205lr"
"python -O toySeg_cons.py seed=123 lr=0.005 weight_epoch=100 ratio=0.3 save_dir=${save_dir}/symmetry_lower_30_205lr"
"python -O toySeg_cons.py seed=123 lr=0.005 weight_epoch=100 ratio=0.5 save_dir=${save_dir}/symmetry_lower_50_205lr"

# cons
"python -O toySeg_cons.py seed=123 train_mode=cons_unlab lr=0.001 weight_epoch=15 ratio=0.3 weights.cons_weight=0.5 weights.weight_cons=0.005 save_dir=${save_dir}/symmetry_05advcons_205cons "
"python -O toySeg_cons.py seed=123 train_mode=cons_unlab lr=0.001 weight_epoch=15 ratio=0.3 weights.cons_weight=0.5 weights.weight_cons=0.0005 save_dir=${save_dir}/symmetry_05advcons_305cons "

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

