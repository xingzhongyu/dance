# export OPENAI_API_KEY=AIzaSyAaSHIyeerW4155sTd6GwYdHiuBgYctJzM
export OPENAI_API_KEY=sk-acad184f8a7848918d7c9046affd1b65
export http_proxy=http://121.250.209.147:7890
export https_proxy=http://121.250.209.147:7890

python initial_program.py --num_runs 1 --n_epochs 10 --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743 --device cuda:5
python initial_program.py --num_runs 1 --n_epochs 10 --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --device cuda:5
python initial_program.py --num_runs 1 --n_epochs 10 --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:5
python initial_program.py --num_runs 1 --n_epochs 10 --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --device cuda:5