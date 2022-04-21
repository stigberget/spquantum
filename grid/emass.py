
from warnings import WarningMessage
import numpy as np 
import matplotlib.pyplot as plt
from utils.converters import convert_length, convert_mass

class EffectiveMassBase:

    def __init__(self,mesh,unit,label):
        
        self.mesh = mesh
        self.tensor = None
        self.unit = unit
        self._label = label

    def get_label(self):
        return self._label

    def _isaligned(self):

        """
        Verifies that the number of nodes corresponds to the number of potential values
        """
        if(self.mesh.num_nodes != self.tensor.shape[0]):
            emsg = "Number of potential values does not match the number of nodes in the mesh"
            raise ValueError(emsg)

class ConstEffectiveMass(EffectiveMassBase):

    def __init__(self,mesh,mstar=None,unit='kg',verbose=0):

        label = 'const'

        super().__init__(mesh,unit,label)

        if(verbose == 1):
            print("\nSetting up effective mass...")


        if(mstar is None):
            return
        else:
            self.tensor = np.zeros((self.mesh.num_nodes,1))
            self.tensor += mstar

    def set_mass(self,mstar):

        if(self.tensor is None):
            self.tensor = np.zeros((self.mesh.num_nodes,1))
            self.tensor += mstar
        else:
            self.tensor -= self.tensor + mstar

    def inspect(self,mass_scale='mel',length_scale='nm'):
        """
        inspect - produces a pyplot instance that plots the effective mass

        Args:
        
        mass_scale - unit used to 

        length_scale - unit 

        """

        if(self.tensor is None):
            emsg = "Warning: effective masses have not been assigned yet"
            raise WarningMessage(emsg)
    
        
        meff,_ = convert_mass(self.tensor,self.unit,target=mass_scale) # Get the desired unit for the effective mass
        x,_ = convert_length(self.mesh.X,self.mesh.scale,target=length_scale) # Get the desired unit for the length dimensions

        plt.plot(x,meff)
        ax = plt.gca()
        ax.set_xlabel(f"z-coordinate [{length_scale}]")
        ax.set_ylabel(f"Effective Mass [{mass_scale}]")
        ax.set_xlim([np.min(x),np.max(x)])
        ax.grid(True,which='both')
        plt.show()


class EffectiveMass1D(EffectiveMassBase):

    def __init__(self,mesh,unit='kg',verbose=0):

        label = 'variable'

        if(verbose == 1):
            print("\n Setting up 1D effective mass...")

        super().__init__(mesh,unit,label)

    def set_mass(self,effective_mass_fcn,args):

        self.tensor = effective_mass_fcn(*args)
    
    def link(self,tensor):
        """
        Method link:



        """

        self.tensor = tensor

    def inspect(self,mass_scale='mel',length_scale='nm'):
        """
        Method inspect:
        produces a pyplot instance that plots the effective mass over spatial coordinates

        Args:
        
        mass_scale - unit used to 

        length_scale - unit 

        """

        if(self.tensor is None):
            emsg = "Warning: effective masses have not been assigned yet"
            raise WarningMessage(emsg)
            
        
        meff,_ = convert_mass(self.tensor,self.unit,target=mass_scale) # Get the desired unit for the effective mass
        x,_ = convert_length(self.mesh.X,self.mesh.scale,target=length_scale) # Get the desired unit for the length dimensions

        plt.plot(x,meff)
        ax = plt.gca()
        ax.set_xlabel(f"z-coordinate [{length_scale}]")
        ax.set_ylabel(f"Effective Mass [{mass_scale}]")
        ax.set_xlim([np.min(x),np.max(x)])
        ax.grid(True,which='both')
        plt.show()

    



    