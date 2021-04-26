FROM ubuntu:18.04

RUN apt-get update
RUN set -ex; \
	apt-get install python3.6 -y; \
	apt-get install python3-pip -y; \
	python3.6 -m pip install --upgrade virtualenv; \
	apt-get install python3-venv -y
RUN set -ex; \
	apt install git -y; \
	git clone https://www.github.com/calvinkrist/alloxtesting.git; \
	cd alloxtesting/estimator; \
	git fetch; \
	git checkout addExperiments; \
	python3.6 -m venv .venv; \
	. .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install numpy; \
	pip install scipy; \
	pip install tensorflow==1.15; \
	pip install tensorflow-gpu==1.15; \
	pip install scikit-image; \
	pip install pandas==0.24
COPY base.sh /base.sh