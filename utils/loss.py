from abc import ABC, abstractmethod
import torch


class DenoisingLoss(ABC):
    @abstractmethod
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Base class for denoising loss.
        Input:
            - x: the clean data with shape [B, F, C, H, W]
            - x_pred: the predicted clean data with shape [B, F, C, H, W]
            - noise: the noise with shape [B, F, C, H, W]
            - noise_pred: the predicted noise with shape [B, F, C, H, W]
            - alphas_cumprod: the cumulative product of alphas (defining the noise schedule) with shape [T]
            - timestep: the current timestep with shape [B, F]
        """
        pass


class X0PredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((x - x_pred) ** 2)


class VPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        weights = 1 / (1 - alphas_cumprod[timestep].reshape(*timestep.shape, 1, 1, 1))
        return torch.mean(weights * (x - x_pred) ** 2)


class NoisePredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((noise - noise_pred) ** 2)


class FlowPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((kwargs["flow_pred"] - (noise - x)) ** 2)


NAME_TO_CLASS = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss
}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]
