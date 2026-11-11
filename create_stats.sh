echo "Collecting SASS stats... outputting to sass_stats.csv"
./util/job_launching/run_simulations.py -C QV100-SASS-1B_INSN -B ece695.part1 -T /home/min/a/ee695paa/accel-sim-traces/rodinia-3.1/11.2/ -N sass-run -c 10
./util/job_launching/monitor_func_test.py -N sass-run -v
./util/job_launching/get_stats.py -C QV100-SASS-1B_INSN,QV100-PTX-1B_INSN -B ece695.part1 -R | tee sass_stats.csv
