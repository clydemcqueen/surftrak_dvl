./replay_terrain.py --csv data/transect1.tlog
./replay_terrain.py --csv data/transect2.tlog
./replay_terrain.py --csv data/short.tlog

./graph_results.py data/transect1_TEKF.csv
./graph_results.py data/transect2_TEKF.csv
./graph_results.py --zoom 1759948600 data/short_TEKF.csv
