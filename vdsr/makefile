TIMESTAMP=$(shell date +%Y%m%d%H%M%S)
NAME_JOB=vdsr_$(TIMESTAMP)
PATH_CHECKPOINT=gs://hecate-research-ml-results/$(NAME_JOB)_ckpt/
PATH_LOG=gs://hecate-research-ml-results/$(NAME_JOB)_log/

deploy:
	gcloud ml-engine jobs submit training $(NAME_JOB) \
		--module-name vdsr.experiment_train \
		--package-path vdsr \
		--python-version=3.5 \
		--staging-bucket gs://hecate-research-ml-staging/ \
		--region asia-east1 \
		--runtime-version=1.8 \
		--scale-tier=CUSTOM \
		--config=config.yaml \
		-- \
		--ckpt_path=$(PATH_CHECKPOINT) \
		--logs_path=$(PATH_LOG) \
		--data_path=gs://hecate-research-datasets/vdsr_291/291/ \
		--scaling_factors=1.5/2.0/2.5/3.0/3.5/4.0 \
		--save_checkpoint_steps=1280 \
		--image_size=41 \
		--batch_size=64 \
		--num_layers=20 \
		--initial_learning_rate=0.1 \
		--learning_rate_decay_steps=2560 \
		--learning_rate_decay_factor=0.1

