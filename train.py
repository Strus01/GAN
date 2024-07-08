import hydra
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, open_dict

from data.datamodule import AbstractGalleryDataModule
from model.dcgan import DCGAN


@hydra.main(version_base=None, config_path='config', config_name='config')
def run(config: DictConfig) -> None:
    seed_everything(123)

    with open_dict(config):
        for i in range(len(config.model.generator.conv_blocks)):
            config.model.generator.conv_blocks[i] = list(eval(config.model.generator.conv_blocks[i]))
        for i in range(len(config.model.discriminator.conv_blocks)):
            config.model.discriminator.conv_blocks[i] = list(eval(config.model.discriminator.conv_blocks[i]))

    dm = AbstractGalleryDataModule(config.dataset)

    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = TensorBoardLogger(save_dir=hydra_output_path)

    trainer = Trainer(
        **config.train.trainer,
        default_root_dir=hydra_output_path,
        enable_checkpointing=True,
        logger=logger
    )

    model = DCGAN(config)
    trainer.fit(model, dm, ckpt_path=config.train.ckpt_path)


if __name__ == '__main__':
    run()
