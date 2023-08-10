# docker build -t goma-quri . --build-arg NB_UID=`id -u` && docker run --rm -it -v $PWD:/work -w /work goma-quri bash run_in_docker.sh
poetry install
poetry run pip3 install openfermionpyscf
poetry run python hweansatz_benchmark.py
