from train_CurriculumLearning import main
from training.common import run_train_entrypoint


if __name__ == "__main__":
    run_train_entrypoint(main, default_config="configs/config_cta.yaml")
