NAME=videomamba
COMMAND?=bash
OPTIONS?=

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

.PHONY: all
all: stop build run

.PHONY: build
build:
	docker build --target final -t $(NAME) .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: run
run:
	docker run --rm -dit \
		--net=host \
		--ipc=host \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		--user $(USER_ID):$(GROUP_ID) \
		-v $(shell pwd):/workspace \
		--name=$(NAME) \
		$(NAME) \
		$(COMMAND)

	docker attach $(NAME)


.PHONY: attach
attach:
	docker attach $(NAME)


.PHONY: notebook
notebook:
	docker run --rm -it \
		--ipc=host \
		-p 8888:8888 \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		--user $(USER_ID):$(GROUP_ID) \
		-v $(shell pwd):/workspace \
		--name=$(NAME)_notebook \
		$(NAME) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace
