#!/bin/bash

dataset_name=(600.perlbench_s 602.gcc_s 605.mcf_s 607.cactuBSSN_s 620.omnetpp_s 621.wrf_s 623.xalancbmk_s 625.x264_s 627.cam4_s 638.imagick_s 641.leela_s 644.nab_s 648.exchange2_s 649.fotonik3d_s 654.roms_s 996.specrand_fs 998.specrand_is)


moo_list=(cpi-power cpi-area)

for dataset in "${dataset_name[@]}"; do
    for moo in "${moo_list[@]}"; do
        python main.py --dataset="$dataset" --moo="$moo"
        echo "$dataset for $moo has done."
    done
done