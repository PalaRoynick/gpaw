from gpaw.extensions.dispersion import DN

class D3(DN):
    name = 'd3'
    def __init__(self, *, xc, **kwargs):
        super().__init__(xc=xc, **kwargs)
