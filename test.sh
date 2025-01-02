for num in 100 200 400 1000 2000 4000 10000 20000 40000 100000 200000 400000 1000000; do
    for datapoint in 4 8 16 32; do
        for time in 1; do
            /home/kelvin/anaconda3/envs/evogp312/bin/python /home/kelvin/test/evogp/sr_test.py $num $datapoint $time
        done
    done
done