from torch.utils.data import DataLoader

def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Calculate mean and variance
    mean = 0.
    var = 0.
    nb_samples = 0.

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        var += images.var(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    var /= nb_samples
    std = var ** 0.5
    return mean, std