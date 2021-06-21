for tp in "313 3866" "333 3541" "353 3158" "373 2739"
do
    set -- $tp
    scp charlesli@pod-login1.cnsi.ucsb.edu:~/srel/rand_copolymer/25A4_25A12_rand1_10wt/NPT_${1}K_${2}bar/srel3_smear1.0/system_ff.dat 25A4_25A12_NPT_${1}K_${2}bar_10wt_ff.dat
done
