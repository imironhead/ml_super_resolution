MODEL=pat
TIMESTAMP=$(shell date +%Y%m%d%H%M%S)
NAME_JOB=enet_$(TIMESTAMP)_$(MODEL)
PATH_CHECKPOINT=gs://hecate-research-ml-results/$(NAME_JOB)_ckpt/
PATH_LOG=gs://hecate-research-ml-results/$(NAME_JOB)_log/

train:
	gcloud ml-engine jobs submit training $(NAME_JOB) \
		--module-name enet.experiment_train \
		--package-path enet \
		--python-version=3.5 \
		--staging-bucket gs://hecate-research-ml-staging/ \
		--region asia-east1 \
		--runtime-version=1.8 \
		--scale-tier=CUSTOM \
		--config=config.yaml \
		-- \
		--train_dir_path=gs://hecate-research-datasets/mscoco2014/train2014_256x256/ \
		--vgg19_path=gs://hecate-research-model-weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.npz \
		--ckpt_path=$(PATH_CHECKPOINT) \
		--log_path=$(PATH_LOG) \
		--model=$(MODEL)

