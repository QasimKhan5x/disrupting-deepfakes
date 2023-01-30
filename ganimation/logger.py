from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def image_summary(self, name, x, step):
        self.writer.add_images(tag=name, img_tensor=x, global_step=step)
        
