class NeedsGD:
    """Common base class for classes using a grid descriptor."""

    # XXX/CB: We might be adding to this with pd2/3...
    def __init__(self):
        self.gd = None

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def allocate(self):
        pass

    def estimate_memory(self, mem):
        raise NotImplementedError
