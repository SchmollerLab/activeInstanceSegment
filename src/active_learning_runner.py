
from config_builder import get_config
from active_learning.active_learning_trainer import ActiveLearningTrainer


if __name__ == "__main__":
    
    cfg = get_config("al_pipeline_config")
    al_trainer = ActiveLearningTrainer(cfg, is_test_mode=False)
    al_trainer.run()
