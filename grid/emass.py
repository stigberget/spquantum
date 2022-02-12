import numpy as np 


class EffectiveMassBase:

    def __init__(self,Grid):
        
        self.Grid = Grid
        self.mstar = None

    def set_emass(self,mstar):

        self.mstar = mstar


class ConstEffectiveMass(EffectiveMassBase):

    def __init__(self,Grid,mstar=None):

        super().__init__(Grid)

        self.mstar = mstar

    

class EffectiveMass(EffectiveMassBase):

    def __init__(self):

        super().__init__()

    def setmass(self,effective_mass_fcn,args):

        self.mstar = effective_mass_fcn(*args)

    



    