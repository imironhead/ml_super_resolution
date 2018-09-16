TIMESTAMP=$(shell date +%Y%m%d%H%M%S)
BUCKET_NAME=oooxxx-research
BUCKET_STAGING=gs://$(BUCKET_NAME)-ml-staging/
BUCKET_RESULTS=gs://$(BUCKET_NAME)-ml-results/
BUCKET_DATASETS=gs://$(BUCKET_NAME)-datasets/

PROJECT_NAME=espcn

SCALING_FACTOR=3
JOB_NAME=$(PROJECT_NAME)_$(SCALING_FACTOR)x_$(TIMESTAMP)

DATA_PATH=$(BUCKET_DATASETS)$(PROJECT_NAME)/91/x$(SCALING_FACTOR)/
CKPT_PATH=$(BUCKET_RESULTS)$(PROJECT_NAME)/ckpt/$(JOB_NAME)/
LOGS_PATH=$(BUCKET_RESULTS)$(PROJECT_NAME)/logs/$(JOB_NAME)/

train:
	gcloud ml-engine jobs submit training $(JOB_NAME) \
		--module-name=espcn.experiment_train \
		--package-path=espcn \
		--python-version=3.5 \
		--staging-bucket=$(BUCKET_STAGING) \
		--region=asia-east1 \
		--runtime-version=1.8 \
		--scale-tier=CUSTOM \
		--config=config.yaml \
		-- \
		--data_path=$(DATA_PATH) \
		--ckpt_path=$(CKPT_PATH) \
		--logs_path=$(LOGS_PATH) \
		--batch_size=64 \
		--scaling_factor=$(SCALING_FACTOR) \
		--lr_patch_size=17 \
		--initial_learning_rate=0.01 \
		--learning_rate_decay_factor=0.95 \
		--learning_rate_decay_steps=8000 \
		--stop_training_at_k_step=1600000

