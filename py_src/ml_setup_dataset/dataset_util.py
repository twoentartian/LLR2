from typing import cast
from torch import Tensor
from torch.utils.data import DataLoader

def calculate_mean_std(dataset) -> tuple[Tensor, Tensor]:
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    mean: Tensor | None = None
    var: Tensor | None = None
    nb_samples = 0

    for images, _ in loader:
        assert isinstance(images, Tensor)

        batch_samples = images.size(0)
        images = images.reshape(batch_samples, images.size(1), -1)

        batch_mean = images.mean(dim=2).sum(dim=0)
        batch_var = images.var(dim=2).sum(dim=0)

        mean = batch_mean if mean is None else mean + batch_mean
        var = batch_var if var is None else var + batch_var
        nb_samples += batch_samples

    assert mean is not None
    assert var is not None

    mean = cast(Tensor, mean / nb_samples)
    std = cast(Tensor, (var / nb_samples).sqrt())
    return mean, std
