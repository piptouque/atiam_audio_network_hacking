import importlib
from datetime import datetime


class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        succeeded = False
        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = self._get_ext_writer(log_dir)

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)
        self._enabled = enabled and succeeded

        self.step = 0
        self.mode = ''

        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()
    
    def _get_ext_writer(self, log_dir: str) -> bool:
        succeeded = False
        for module in ["torch.utils.tensorboard", "tensorboardX"]:
            try:
                writer = importlib.import_module(module).SummaryWriter(log_dir)
                succeeded = True
            except ImportError:
                succeeded = False
            if succeeded:
                self.writer = writer
                self.selected_module = module
                break
        return succeeded

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if self._enabled:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
