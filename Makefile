.PHONY: train watch TensorBoard

train:
	python train_teacher_curriculum_v2.py

watch:
	python watch_teacher_visual.py

TensorBoard:
	tensorboard --logdir ./logs/teacher_curriculum_v7/ --host 0.0.0.0