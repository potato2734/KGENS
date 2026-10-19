bash run.sh train RotatE FB15k237 0 0 512 256 1000 9 2.2 0.00004 100000 16 0.5 0.1 0.0 0.0 "" -de
bash run.sh train RotatE wn18rr 0 0 256 256 1000 6 5.0 0.0001 100000 8 3.0 1.0 0.0 0.0 "" -de
bash run.sh train RotatE YAGO3-10 0 0 1024 512 1000 24 1.0 0.0003 100000 4 0.3 1.0 0.0 0.0 "" -de 

bash run.sh train TransE FB15k237 0 0 256 512 1000 6 2.0 0.0001 100000 8 0.7 0.0 0.0 0.0 ""
bash run.sh train TransE wn18rr 0 0 512 512 250 4 6.0 0.0007 100000 8 6.0 0.2 0.0 0.0 "" 
bash run.sh train TransE YAGO3-10 0 0 1024 1024 1000 28 0.6 0.0002 100000 8 0.6 1.1 0.0 0.0 "" 

bash run.sh train ComplEx FB15k237 0 0 512 512 1000 200.0 2.0 0.0006 100000 16 1.0 0.5 0.0 1.0 "" -de -dr -r 0.000007 --ignore_scoring_margin 
bash run.sh train ComplEx wn18rr 0 0 512 64 1000 200.0 2.0 0.0009 100000 8 0.8 1.0 0.0 0.0 "" -de -dr -r 0.000005 --ignore_scoring_margin
bash run.sh train ComplEx YAGO3-10 0 0 1024 1400 500 200.0 1.5 0.0003 100000 8 0.6 1.1 0.0 0.3 "" -de -dr -r 0.0000012 --ignore_scoring_margin

bash run.sh train DistMult FB15k237 0 0 1024 256 1000 200.0 2.0 0.0009 100000 16 0.7 0.8 0.0 1.0 "" -r 0.000004 --ignore_scoring_margin 
bash run.sh train DistMult wn18rr 0 0 512 32 1000 200.0 2.0 0.0014 100000 8 1.0 1.0 0.0 0.8 "" -r 0.000003 --ignore_scoring_margin 
bash run.sh train DistMult YAGO3-10 0 0 1024 2000 1000 200.0 3.0 0.0003 100000 8 0.6 1.0 0.0 0.2 "" -r 0.000002 --ignore_scoring_margin

bash run.sh train RotatE tree15 0 0 512 1024 1000 6 5.0 0.0001 100000 8 1.0 0.6 0.0 0.2 "" -de
bash run.sh train TransE tree15 0 0 1024 512 1000 4 1.5 0.0002 100000 8 0.7 0.1 0.0 0.0 ""
bash run.sh train ComplEx tree15 0 0 512 128 500 200.0 3.0 0.0001 100000 8 1.0 1.0 0.0 0.0 "" -de -dr -r 0.000003 --ignore_scoring_margin
bash run.sh train DistMult tree15 0 0 512 128 125 200.0 3.0 0.0008 100000 8 1.0 0.6 0.0 0.6 "" -r 0.000005 --ignore_scoring_margin