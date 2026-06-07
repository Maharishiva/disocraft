# Модель обучалась в x-space: на вход dynamics подается зашумленный latent
# x_tau = tau * x1 + (1 - c * tau) * noise, где c = 1 - eps
# и модель возвращает `x_pred ~= x1`, то есть предсказание чистого latent.
# Ниже код семплинга с багом. Исправьте его.


import torch

@torch.no_grad()
def sample_xpred_flow(
    model,
    x: torch.Tensor,              # [B, N, D], initial noise latent
    actions: torch.Tensor,        # [B] or [B, ...]
    num_steps: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    model(x, actions, tau) -> x_pred [B, N, D]
    tau: [B, 1, 1], float in [0, 1)
    """
    dt = 1.0 / num_steps

    for k in range(num_steps):
        tau = torch.full(
            (x.shape[0], 1, 1),
            k / num_steps,
            device=x.device,
            dtype=x.dtype,
        )
        x_pred = model(x, actions, tau)
        c = 1.0 - eps
        x = x + dt * (x_pred - c * x) / (1.0 - c * tau)

    return x
