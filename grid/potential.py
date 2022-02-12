import numpy as np
import matplotlib.pyplot as plt
import utils.converters as cnvtr

class Potential: 
    def __init__(self,Grid): 
        print("Initializing potential...")

        self.Grid = Grid
        self.V = None
    
    def inspect(self):
        
        fig,ax = plt.subplots()

        ax.plot(self.Grid.X,self.V)

    def stats(self):
        print("Potential statistics")


class Potential1D(Potential):

    def __init__(self,Grid):

        if(Grid.ndims != 1):
            emsg = f"A 1D potential was initialized, but the specified mesh is {Grid.ndims}D"
            raise RuntimeError(emsg)

        super().__init__(Grid)

    def generate(self,potential_function):
        self.V = potential_function(self.Grid.X)

    def linear_potential(self,z,V0,V1,layers,mstar,deltaV,units):

        Nz = self.Grid.ndims

        if(units['V'] != 'V'):
            V0,_ = cnvtr.convert_potential(V0,units)
            V1,_ = cnvtr.convert_potential(V1,units)
        if(units['layer'] != 'm'):
            layers,_ = cnvtr.convert_length(layers,units)
        if(units['mstar'] != 'kg'):
            mstar,_ = cnvtr.convert_mass(mstar,units)
        

        wlayers = np.cumsum(layers)
        layerID = 0

        dVdz = (V1 - V0)/(z[-1][0] - z[0][0])

        self.Vz = np.zeros(Nz,1)


        for i in range(Nz):

            if(wlayers[layerID] <= z[i][0]):
                self.Vz[i][0] = self.Vz[i][0] + deltaV
                layerID += 1
            
        return         

    def set_potential(self,V):
        self.V = V

        self._isaligned()

    def _isaligned(self):

        """
        Verifies that the number of nodes corresponds to the number of potential
        """

        if(self.Grid.Nnodes == self.V.shape[0]):
            flag = True
        else:
            flag = False

        return flag




class Potential2D(Potential):

    def __init__(self,Grid):

        raise NotImplementedError("2D potential not implemented yet")

        if(Grid.ndims != 2):
            emsg = f"A 2D potential was initialized, but the specified mesh is {Grid.ndims}D"
            raise RuntimeError(emsg)


class Potential3D(Potential):

    def __init__(self,Grid):

        raise NotImplementedError("3D potential not implemented yet")

        if(Grid.ndims != 3):
            emsg = f"A 3D potential was initialized, but the specified mesh is {Grid.ndims}D"
            raise RuntimeError(emsg)
    



        
        
