TIMESTAMP=$(shell date +%Y%m%d%H%M%S)
SCALING_FACTORS=2_3_4
NAME_JOB=vdsr_$(SCALING_FACTORS)_$(TIMESTAMP)
PATH_CHECKPOINT=gs://hecate-research-ml-results/ckpt_vdsr/$(NAME_JOB)/
PATH_LOG=gs://hecate-research-ml-results/log_vdsr/$(NAME_JOB)/

submit:
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
		--scaling_factors=$(SCALING_FACTORS) \
		--stop_training_at_k_step=25600 \
		--image_size=128 \
		--batch_size=64 \
		--num_layers=20 \
		--use_adam \
		--initial_learning_rate=0.00001 \
		--learning_rate_decay_steps=2560 \
		--learning_rate_decay_factor=0.9

