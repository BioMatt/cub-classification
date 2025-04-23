import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule
import optuna # Hyperparameter tuning module
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#def seed_everything(seed): # This is a method to fix the random seed for everything I'm using to one seed for reproducibility
    #random.seed(seed)
    #numpy.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.man

# Better yet, use pytorch lightning to seed everything in one line. Except optuna
pl.seed_everything(42)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    classification_weight = trial.suggest_float("classification_weight", 0.1, 1.)
    regression_weight = trial.suggest_float("regression_weight", 0.1, 1.)

    wandb_logger = WandbLogger(
        project="CUB-Regression-Classification",
        #name=f'{args.classification_weight}-{args.regression_weight}-{args.lr}',
        log_model=True,
        save_dir='reports',
        name=f"trial-{trial.number}"
    )

    wandb_logger.experiment.config.update({
        "lr":lr,
        "classification_weight":classification_weight,
        "regression_weight":regression_weight
    })

    wandb_logger.experiment.config.update(
        {
            'classification_weight':classification_weight,
            'regression_weight':regression_weight,
            'learning_rate':lr
        }
    )

    data_module = CUBDataModule(
        data_dir=Path(args.data_dir),
        batch_size=4
    )

    data_module.setup()

    model = CUBModel(
        num_classes=200,
        #train_classification=args.train_classification,
       # train_regression=args.train_regression,
        classification_weight=classification_weight,
        regression_weight=regression_weight
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_combined_metric',
        patience=2,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=5, logger=wandb_logger, callbacks=[early_stopping_callback],
                         precision='16-mixed')

    trainer.fit(model, datamodule=data_module)

    wandb_logger.experiment.finish()

    return trainer.callback_metrics["val_combined_metric"].item()


if __name__=="__main__": # Check if this is the main execution of the script before importing. 
                         # So only run the stuff here is it is the main script run and not imported by another
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--n_trials", type=int, default=20)
    #parser.add_argument("--train-classification", type=bool, default=True)
    #parser.add_argument("--train-regression", type=bool, default=True)
    #parser.add_argument("--classification-weight", type=float, default=1.0)
    #parser.add_argument("--regression-weight", type=float, default=1.0)
    #parser.add_argument("--lr", type=float, default=1e-3)
    #  Only using these arguments here, but could add learning rate, number of epochs, output dir, etc.

    args = parser.parse_args()

    study = optuna.create_study(
        direction='maximize',
        study_name='CUB-Class-and-Regr',
        storage="sqlite:///cub_optuna_study.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=args.n_trials)
