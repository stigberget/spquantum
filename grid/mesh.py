
import numpy as np 
import utils.const as const
import utils.converters as cnvtr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Mesh: 

    def __init__(self,name,ndims):

        self.name = name

        self.ndims = ndims

        self.Nx = int(0)
        self.Ny = int(0)
        self.Nz = int(0)

        self.X = None
        self.Y = None
        self.Z = None

        self.scale = 'm'

        self.N = np.array([self.Nx,self.Ny,self.Nx])


    def load_mesh(self,coords):
        """
        Method to load an active or cached numpy array into the mesh object

        Args:

        coords - Numpy array of size mesh.ndims x num_nodes that describes the mesh
        """

        if(coords.shape[0] != self.ndims):
            raise AssertionError(f"Index 0 of passed arrray does not correspond to dimension of specified mesh object")
        
        self.node_coords = coords

    def mesh_from_file(self,fname):

        """
        Method to load mesh coordinates from the file

        Args:
        
        fname: Specify the path to the file containing the mesh
        """

        raise NotImplementedError()

    def inspect(self,length_scale='default'):

        """
        Visually inspect mesh

        Args:

        length_scale - specify units to be displayed
        """
        if(self.X is None):
            emsg = "Mesh has not yet been generated" 
            raise ValueError(emsg)
        
        if(length_scale == 'default'):
            length_scale = 'nm'

        axlabels = []
        
        if(self.ndims >= 1):
            X,_ = cnvtr.convert_length(self.X,unit=self.scale,target=length_scale)
            axlabels.append(f"x-coordinate [{length_scale}]")
        if(self.ndims >= 2):
            Y,_ = cnvtr.convert_length(self.Y,unit=self.scale,target=length_scale)
            axlabels.append(f"y-coordinate [{length_scale}]")
        if(self.ndims >= 3):
            Z,_ = cnvtr.convert_length(self.Z,unit=self.scale,target=length_scale)
            axlabels.append(f"z-coordinate [{length_scale}]")

        fig,ax = plt.subplots()

        
            

        if(self.ndims==1):
            Ytmp = np.zeros((X.shape[0],1))
            ax.scatter(X,Ytmp)
            ax.plot(X,Ytmp)
            ax.set_xlabel(axlabels[0])
            ax.set_xlim(xmin=np.min(X),xmax=np.max(X))
            ax.grid(True,'both')
        elif(self.ndims==2):
            ax.scatter(X,Y)
            hsegs = np.stack((X,Y), axis=2)
            vsegs = hsegs.transpose(1,0,2)
            ax.add_collection(LineCollection(hsegs))
            ax.add_collection(LineCollection(vsegs))
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[1])
        elif(self.ndims==3):
            raise NotImplementedError()
            #TODO: Configure 3D plotting
            ax.axes(projections='3d')
            ax.wire
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[0])
            ax.set_

        fig.show()

    def save_mesh(self,fname):
        NotImplementedError

    
    def stats(self):

        divisor = "----------------------------------------"

        print(f"Mesh Statistics")
        print(divisor)
        print(str.ljust("Mesh dimensions",20),"|",str.rjust(f"{self.ndims}",15))
        print(str.ljust("Number of nodes",20),"|",str.rjust(f"{np.sum(self.N)}",15))

    def _updateN(self):
        self.N = [self.Nx,self.Ny,self.Nz]


        



class Mesh1D(Mesh): 

    def __init__(self,name):

        super().__init__(name=name,ndims=1)

    def generate(self,x1,x2,N,unit=None):

        self.X = np.linspace(x1,x2,N)
        self.Nx = N

        if(unit != 'm'):
            self.X,self.scale = cnvtr.convert_length(self.X,unit)

        self._updateN()
    
    def generate_skew(self,z1,z2,unit,N,skewness):
        raise NotImplementedError()
    
 
class Mesh2D: 

    def __init__(self):
        self.X = None
        self.Y = None

        print("Initializing 2D mesh...")
    
    def square(self,x1,x2,y1,y2,Nx,Ny,units=None):

        if(units == None):
            units['x'] = 'nm'
            units['y'] = 'nm'

        x = np.linspace(x1,x2,Nx)
        y = np.linspace(y1,y2,Ny)

        if(units['x'] != 'm'):
            x,_ = cnvtr.convert_length(x,units['x'])
        if(units['y'] != 'm'):
            y,_ = cnvtr.convert_length(y,units['y'])

        

        self.X, self.Y = np.meshgrid(x,y)

        




    


    
    
        


        