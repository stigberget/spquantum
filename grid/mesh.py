import numpy as np 
import utils.const as const
import utils.converters as cnvtr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import math

class Mesh: 

    def __init__(self,name,ndims):

        self.name = name

        self.ndims = ndims
        
        self.num_nodes = int(0)

        self.nx = int(0)
        self.ny = int(0)
        self.nz = int(0)

        self.coords = None

        self.X = None
        self.Y = None
        self.Z = None

        self.adjacency = None

        self.scale = 'm'


    def load_mesh(self,coords=None,X=None,Y=None,Z=None):
        """
        Method to load an active or cached numpy array into the mesh object

        Args:

        coords - Numpy array of size mesh.ndims x num_nodes that describes the mesh
        """

        if(coords.shape[0] != self.ndims or coords.shape[1] != self.ndims):
            raise AssertionError(f"Index 0 of passed arrray does not correspond to dimension of specified mesh object")
        
        if(X is None):
            if(self.ndims == 1):
                self.coords = coords
                self.X = coords[0,:]
            if(self.ndims == 2):
                self.Y = coords[1,:]
            if(self.ndims == 3):
                self.Z = coords[2,:]
        else:
            if(self.ndims == 1):
                self.X = X
                self.coords = X
            if(self.ndims == 2):
                self.Y = Y
                self.coords = np.vstack([self.coords,Y])
            if(self.ndims == 3):
                self.Z = Z
                self.coords = np.vstack([self.coords,Z])

    def mesh_from_file(self,fname):

        """
        Method to load mesh coordinates from the file

        Args:
        
        fname: Specify the path to the file containing the mesh
        """

        raise NotImplementedError()

    def inspect(self,length_scale='default'):

        """
        inspect - Visually inspect mesh

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
            hsegs = np.stack((X,Y), axis=1)
            vsegs = hsegs.transpose(1,0,2)
            ax.add_collection(LineCollection(hsegs))
            ax.add_collection(LineCollection(vsegs))
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[1])
        elif(self.ndims==3):
            raise NotImplementedError()
            #TODO: Configure 3D plotting
            ax.axes(projections='3d')
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[1])
            ax.set_zlabel(axlabels[2])

        fig.show()

    def save_mesh(self,fname):
        raise NotImplementedError()

    
    def stats(self):

        divisor = "----------------------------------------"
        print(f"Mesh Statistics")
        print(divisor)
        print(str.ljust("Mesh dimensions",20),"|",str.rjust(f"{self.ndims}",15))
        print(str.ljust("Number of nodes",20),"|",str.rjust(f"{np.sum(self.num_nodes)}",15))

    def get_name(self):
        return self.name


        



class Mesh1D(Mesh): 

    def __init__(self,name,verbose=0):

        super().__init__(name=name,ndims=1)

        if(verbose == 1):
            print("\nIntializing 1D mesh...")
            

    def generate(self,x1,x2,num_nodes,unit='nm'):

        """
        generate - generates a uniform mesh with uniform spacing and stores it into the Mesh1D.coords instance variable referenced
        by the Mesh1D.X instance variable. 

        Args:

        x1: int float double specifying x-coordinate of the edge node 

        x2: int float double specifying x-coordinate of the edge node

        num_nodes: int specifying the number of equally spaced nodes (including x1 and x2) 
        """

        self.X = np.linspace(x1,x2,num_nodes)

        self.num_nodes = num_nodes

        self.nx = num_nodes

        if(unit != 'm'):
            self.X,self.scale = cnvtr.convert_length(self.X,unit)

    
    def generate_skew(self,z1,z2,unit,N,skewness):
        raise NotImplementedError()

    def delta(self,node,degree=1):

        """
        delta - computes the weighted spatial difference delta_x for a given node. If the mesh is uniform (ie. equal distances
        between all nodes) then the function simply returns the distance between two nodes

        Args:

        node:
        int specifying the index of the node where the spatial variation delta_x is to be computed. node arguemnt 
        must reside within the range -1 < node < nx

        degree:
        int specifying how many nodes (in each direction) in the neighborhood of the specified node that are considered. 
        The distance between each node is weighted using nCk weighting. nCk weighting means that the weights are computed using
        Pascal's triangle, and then subsequently multiplied by the difference tensor

        """
        npts = degree + 1

        qpoints = np.linspace(0,degree,npts,dtype='int') # query points
        ppoints = node + qpoints # Get all the nodes in the positive direction including the currently considered node
        rpoints = node - qpoints[1:] # Get all the nodes in the reverse (negative) direction

        pxq = self.X[ppoints[ppoints < self.nx]] # query coordinates & ensure that the node id is not larger than the number of mesh nodes
        rxq = self.X[rpoints[rpoints >= 0]] # query coordinates & ensure that the node id is larger than zero

        xq = np.concatenate([rxq,pxq])

        weight = nck_weights(degree) # get the weight matrix

        dxq = np.diff(xq)
        
        dx = np.mean(np.multiply(dxq,weight))

        return dx

        

    
 
class Mesh2D(Mesh): 

    def __init__(self,name="2D Mesh"):
        
        super().__init__(name,ndims=2)
        print("Initializing 2D mesh...")
    
    def quadratic(self,x1,x2,y1,y2,nx,ny,units='default'):

        self.num_nodes = int(nx*ny)
        self.nx = int(nx)
        self.ny = int(ny)

        if(units == 'default'):
            units = {}
            units['x'] = 'nm'
            units['y'] = 'nm'

        x = np.linspace(x1,x2,nx)
        y = np.linspace(y1,y2,ny)

        if(units['x'] != 'm'):
            x,_ = cnvtr.convert_length(x,units['x'])
        if(units['y'] != 'm'):
            y,_ = cnvtr.convert_length(y,units['y'])

        X,Y = np.meshgrid(x,y)

        self.coords = np.vstack([np.reshape(X,(1,nx*ny)),np.reshape(Y,(1,nx*ny))])
        self.X = self.coords[0,:] 
        self.Y = self.coords[1,:]

        self.adjacency = AdjacencyList(self.num_nodes,self.ndims)

        self.adjacency.auto_assemble(self.coords,self.nx,self.ny)

    def delta(self,node,degree=1):
        """
        
        """
        x_neighbor_index = self.adjacency.get_neighbors(node,dim="x",num_neighbors=degree)
        y_neighbor_index = self.adjacency.get_neighbors(node,dim="y",num_neighbors=degree)

        
        
        nx = len(x_neighbor_index)
        ny = len(y_neighbor_index)

        x_neighbor = np.zeros((nx+1,))
        y_neighbor = np.zeros((ny+1,))

        x_neighbor[0] = self.X[node]
        y_neighbor[0] = self.Y[node]

        x_neighbor[1:] = self.X[x_neighbor_index]
        y_neighbor[1:] = self.Y[y_neighbor_index]

        weight = nck_weights(degree) # get the weight matrix

        x_neighbor = np.sort(x_neighbor - x_neighbor[0])
        y_neighbor = np.sort(y_neighbor - y_neighbor[0])

        # We must determine where in the sorted array that our currently considered node resides. This is necessary
        # so that the weight matrix can be modified, in case the node is on the edge of the computational domain
        # such that the weight matrix first of all has the correct size, and second of all that we are weighting
        # the spatial differences appropriately.

        x_node_index = int(np.argwhere(x_neighbor == 0.0))
        y_node_index = int(np.argwhere(y_neighbor == 0.0))

        x_num_neighbors = x_neighbor.shape[0]
        y_num_neighbors = y_neighbor.shape[0] 
        
        xw_start = degree - x_node_index
        yw_start = degree - y_node_index
        xw_end = xw_start + (x_num_neighbors-1)
        yw_end = yw_start + (y_num_neighbors-1)

        xweight = weight[xw_start:xw_end]
        yweight = weight[yw_start:yw_end]

        dxq = np.diff(x_neighbor) # Compute the pairwise distance between neighboring nodes in (x-direction)
        dyq = np.diff(y_neighbor) # Compute the pairwise distance between neighboring nodes in (y-direction)

        dx = np.sum(np.multiply(dxq,xweight))/np.sum(xweight)
        dy = np.sum(np.multiply(dyq,yweight))/np.sum(yweight)

        return dx,dy

        
class AdjacencyList:

    def __init__(self,num_nodes,ndims=1):
        
        self.ndims = ndims
        self.neighbors = [] 
        self.num_nodes = num_nodes
    
    def auto_assemble(self,coords,nx,ny=None,nz=None):


        # TODO: This is a computational bottleneck and needs to be revisited (implement vectorization)

        """
        Method auto_assemble: 
        Automatically estimates which nodes are connected (ie neighbors) from auto generated meshes 
        [eg. Mesh3D.hexahedral Mesh2D.quadratic() or Mesh1D.generate()] and creates an adjacency list which contains
        the index of nodes that neighbor each node along the x-, y-, and z-planes.

        Since finite difference methods become increasingly complex on unstructured grids the automatic assembly method 
        only provides support on semi-structured grids. On unstructured grids with more complex geometries
        we would need to resort to finite element approaches

        Args:

        coords:
        A MeshXX.coords instance variable that passes the coordinates of all the mesh nodes to the function 


        nx:
        int specifying number of nodes along the x-axis

        ny:
        int specifying number of nodes along the y-axis (if dimensionality of mesh/problem is 2D)

        nz:
        int specifying number of nodes along the z-axis (if dimensionality of mesh/problme is 3D)

        """

        print("Assembling adjacency list...")

        ndims = coords.shape[0]

        for node in range(self.num_nodes - 1):

            neighborID = {}

            if(ndims >= 1):
                if(node%nx == 0):
                    neighborID["x"] = [node+1]
                elif((node+1)%nx == 0):
                    neighborID["x"] = [node-1]
                else:
                    neighborID["x"] = [node-1,node+1]

            if(ndims >= 2):
                if((node//(nx))%ny == 0):
                    neighborID["y"] = [node+nx]
                elif((node+nx)//(nx)%ny == 0):
                    neighborID["y"] = [node-nx]
                else:
                    neighborID["y"] = [node-nx,node+nx]

            if(ndims == 3):
                if((node)//(nx*ny)%nz == 0):
                    neighborID["z"] = [node+nx*ny]
                elif((node+nx*ny)//(nx*ny)%nz == 0):
                    neighborID["z"] = [node-nx*ny]
                else:
                    neighborID["z"] = [node-nx*ny,node+nx*ny]

            self.neighbors.append(neighborID)

    def assemble(self,neighbor_indices):
        """
        Method assemble: 
        assemble the adjacency list from specified neighbor_indices

        Args:
        
        neighbor_indices:

        A list or numpy array containing the indices of neighbors for each node. Each column must correspond
        to a node.
        """ 

        if(len(neighbor_indices) != self.num_nodes):
            emsg = "Length of list of neighbor indices does not match the number of nodes in the computational domain"
            raise IndexError(emsg)

        raise NotImplementedError()
    
    def get_neighbors(self,node,dim,num_neighbors=1):
        """
        Method get_neighbors:  
        """

        # Get the adjacent nodes to the
        adjacent_nodes = self.neighbors[node][dim].copy()
        
        # This is our iterable list that we want to return
        neighbors = adjacent_nodes

        if(num_neighbors) > 1:
            for offset in range(num_neighbors-1):
                
                num_adjacent_nodes = len(adjacent_nodes)
                
                node_1 = self.neighbors[adjacent_nodes[0]][dim]
                
                if(num_adjacent_nodes == 2):
                    node_2 = self.neighbors[adjacent_nodes[1]][dim]
                   
                adjacent_nodes = [] # Delete the information store 

                for trial_node in node_1:    
                    if(trial_node not in neighbors and trial_node != node):
                        adjacent_nodes.append(trial_node)
            
                if(num_adjacent_nodes == 2):
                    for trial_node in node_2: 
                        if(trial_node not in neighbors and trial_node !=node):
                            adjacent_nodes.append(trial_node)
                
                
                #if(adjacent_nodes):
                neighbors.extend(adjacent_nodes)

                


        
        return neighbors
                     
                    
                    
                    

                

            

        





    def arrange(self):
        """
        Finds the best node numbering to reduce the bandwidth of the Hamiltonian matrix (only necessary for semi-structured grids) 
        """
        raise NotImplementedError()


def nck_weights(degree):

    polyorder = 2*degree # polynomial order

    weights = np.zeros(polyorder)

    for i in range(polyorder):
        weights[i] = math.comb(polyorder-1,i)

    return weights




    


    
    
        


        