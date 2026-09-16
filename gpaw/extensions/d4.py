from gpaw.extensions.dispersion import DN


class D4(DN):
    name = 'd4'

    def __init__(self, *, xc, **kwargs):
        super().__init__(xc=xc, **kwargs)
