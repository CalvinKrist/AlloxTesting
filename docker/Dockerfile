FROM ubuntu:18.04

RUN apt-get update
RUN set -ex; \
	apt-get install python3.6 -y; \
	apt-get install python3-pip -y; \
	python3.6 -m pip install --upgrade virtualenv; \
	apt-get install python3-venv -y; \
	python3.6 -m venv .venv
RUN set -ex; \
	. .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install numpy; \
	pip install scipy; \
	pip install tensorflow==1.15; \
	pip install tensorflow-gpu==1.15; \
	pip install scikit-image; \
	pip install pandas==0.24
RUN git clone https://github/com/calvinkrist/alloxtesting.git