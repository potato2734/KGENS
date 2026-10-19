bash run.sh train RotatE FB15k237 0 0 256 256 1000 4 2.5 0.00002 100000 16 0.0 0.0 0.0 0.1 "[!c1]" -de
bash run.sh train RotatE FB15k237 0 0 512 256 1000 16 0.0 0.00007 100000 16 0.5 0.4 0.0 0.0 "[!c2]" -de
bash run.sh train RotatE wn18rr 0 0 256 32 1000 6 2.0 0.00005 100000 8 0.0 0.0 0.0 0.3 "[!c1]" -de
bash run.sh train RotatE wn18rr 0 0 256 256 1000 12 0.0 0.0001 100000 8 2.0 1.0 0.0 0.0 "[!c2]" -de
bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24 1.0 0.0002 100000 4 0.0 0.0 0.0 0.3 "[!c1]" -de 
bash run.sh train RotatE YAGO3-10 0 0 1024 512 1000 32 0.0 0.0001 100000 4 0.5 0.5 0.0 0.0 "[!c2]" -de 

bash run.sh train TransE FB15k237 0 0 512 512 1000 6 1.0 0.00007 100000 8 0.0 0.0 0.0 0.1 "[!c1]"
bash run.sh train TransE FB15k237 0 0 512 512 1000 15 0.0 0.00009 100000 8 0.9 0.4 0.0 0.0 "[!c2]"
bash run.sh train TransE wn18rr 0 0 512 512 250 4 4.0 0.0005 100000 8 0.0 0.0 0.0 0.2 "[!c1]"
bash run.sh train TransE wn18rr 0 0 512 512 250 12 0.0 0.0007 100000 8 6.0 0.7 0.0 0.0 "[!c2]"
bash run.sh train TransE YAGO3-10 0 0 1024 400 500 24 1.0 0.0003 100000 4 0.0 0.0 0.0 0.3 "[!c1]"
bash run.sh train TransE YAGO3-10 0 0 1024 512 500 28 0.0 0.0002 100000 4 0.5 1.0 0.0 0.0 "[!c2]"

bash run.sh train ComplEx FB15k237 0 0 1024 256 1000 200.0 1.0 0.0008 100000 16 0.0 0.0 0.0 0.8 "[!c1]" -de -dr -r 0.00001 --ignore_scoring_margin
bash run.sh train ComplEx FB15k237 0 0 1024 256 1000 200.0 0.0 0.0002 100000 16 0.7 0.3 0.0 0.0 "[!c2]" -de -dr -r 0.000001 --ignore_scoring_margin
bash run.sh train ComplEx wn18rr 0 0 512 64 1000 200.0 1.0 0.0009 100000 8 0.0 0.0 0.0 1.0 "[!c1]" -de -dr -r 0.000005 --ignore_scoring_margin
bash run.sh train ComplEx wn18rr 0 0 512 64 1000 200.0 0.0 0.0009 100000 8 0.3 1.0 0.0 0.0 "[!c2]" -de -dr -r 0.000005 --ignore_scoring_margin
bash run.sh train ComplEx YAGO3-10 0 0 1024 256 500 200.0 1.0 0.0004 100000 4 0.0 0.0 0.0 1.0 "[!c1]" -de -dr -r 0.000001 --ignore_scoring_margin
bash run.sh train ComplEx YAGO3-10 0 0 1024 512 1000 200.0 0.0 0.0001 100000 4 0.5 1.0 0.0 0.0 "[!c2]" -de -dr -r 0.0 --ignore_scoring_margin

bash run.sh train DistMult FB15k237 0 0 1024 256 2000 200.0 1.0 0.0008 100000 16 0.0 0.0 0.0 0.8 "[!c1]" -r 0.00001 --ignore_scoring_margin
bash run.sh train DistMult FB15k237 0 0 1024 256 1000 200.0 0.0 0.0007 100000 16 0.3 1.0 0.0 0.0 "[!c2]" -r 0.0000009 --ignore_scoring_margin
bash run.sh train DistMult wn18rr 0 0 512 32 1000 200.0 0.5 0.0014 100000 8 0.0 0.0 0.0 0.1 "[!c1]" -r 0.000003 --ignore_scoring_margin
bash run.sh train DistMult wn18rr 0 0 512 32 1000 200.0 2.0 0.0014 100000 8 1.0 1.0 0.0 0.0 "[!c2]" -r 0.000003 --ignore_scoring_margin
bash run.sh train DistMult YAGO3-10 0 0 1024 256 1000 400.0 4.0 0.0003 100000 4 0.0 0.0 0.0 1.0 "[!c1]" -r 0.000001 --ignore_scoring_margin
bash run.sh train DistMult YAGO3-10 0 0 512 256 500 400.0 0.0 0.0004 100000 4 0.5 1.0 0.0 0.0 "[!c2]" -r 0.0 --ignore_scoring_margin