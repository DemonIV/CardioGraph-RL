# AGENT: model_agent
"""
Post-hoc kalibrasyon: Temperature Scaling.
logits / T → softmax — tek parametre, F1 değişmez.
"""
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Tek skaler T ile logit ölçekleme.

    Kullanım::

        scaler = TemperatureScaler()
        T = scaler.calibrate(model, val_loader, device)
        probs = torch.softmax(scaler.scale(logits), dim=-1)
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([initial_temperature], dtype=torch.float32)
        )

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """logits / T — gradient akışı korunur."""
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(self, model, val_loader, device, max_iter: int = 50) -> float:
        """
        Validation NLL'ini minimize ederek T'yi LBFGS ile öğren.
        Model ağırlıkları dondurulur, yalnızca T güncellenir.

        Returns
        -------
        float — optimize edilmiş T değeri
        """
        model.eval()

        # Tüm val logitlerini gradient olmadan topla
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                all_logits.append(logits.cpu())
                all_labels.append(batch.y.squeeze().cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.01, max_iter=max_iter
        )

        def _closure():
            optimizer.zero_grad()
            loss = criterion(self.scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return float(self.temperature.item())
