from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.utils import make_grid
import lightning as pl

from model.modules.generator import Generator
from model.modules.discriminator import Discriminator


class DCGAN(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.automatic_optimization = False

        self.config_model = config.model
        self.config_optimizer = config.train.optimizer
        self.save_hyperparameters(
            OmegaConf.to_container(self.config_model, resolve=True) | OmegaConf.to_container(self.config_optimizer, resolve=True)
        )

        self.generator = Generator(self.config_model.generator)
        self.discriminator = Discriminator(self.config_model.discriminator)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

    def weights_init(self, module: nn.Module) -> None:
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def generator_step(self, generated_images: torch.Tensor, stage: str) -> torch.Tensor:
        y_hat_fake = self.discriminator(generated_images)

        generator_loss = F.binary_cross_entropy(y_hat_fake, torch.ones_like(y_hat_fake))

        self.log(f'{stage}/generator_loss', generator_loss, prog_bar=True, sync_dist=True)
        self.log_images(generated_images, stage)
        return generator_loss

    def discriminator_step(self, real_images: torch.Tensor, generated_images: torch.Tensor, stage: str) -> torch.Tensor:
        y_hat_real = self.discriminator(real_images)
        y_hat_fake = self.discriminator(generated_images.detach())

        real_loss = F.binary_cross_entropy(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = F.binary_cross_entropy(y_hat_fake, torch.zeros_like(y_hat_fake))

        discriminator_loss = (real_loss + fake_loss) / 2
        self.log(f'{stage}/discriminator_loss', discriminator_loss, prog_bar=True, sync_dist=True)
        return discriminator_loss
    
    def generate_images(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.config_model.generator.latent_dim, 1, 1, device=self.device).float()
        return self.generator(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        real_images = batch
        generated_images = self.generate_images(real_images.shape[0])
        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Discriminator
        discriminator_loss = self.discriminator_step(real_images, generated_images, 'train')
        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        # Generator
        generator_loss = self.generator_step(generated_images, 'train')
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        real_images = batch
        with torch.no_grad():
            generated_images = self.generate_images(real_images.shape[0])
            self.discriminator_step(real_images, generated_images, 'valid')
            self.generator_step(generated_images, 'valid')

    def configure_optimizers(self):
        generator_optimizer = optim.AdamW(
            self.generator.parameters(), lr=self.config_optimizer.learning_rate, betas=(self.config_optimizer.b1, self.config_optimizer.b2)
        )
        discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), lr=self.config_optimizer.learning_rate, betas=(self.config_optimizer.b1, self.config_optimizer.b2)
        )
        return [generator_optimizer, discriminator_optimizer], []

    def log_images(self, image_tensor: torch.Tensor, stage: str, num_images: int = 25) -> None:
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        self.logger.experiment.add_image(f'{stage}/images', image_grid, self.current_epoch)
