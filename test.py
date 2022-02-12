import numpy as np
import utils.const as const
import utils.converters as cnvtr
import grid.mesh as mesh
#import solvers.spsolve as spsolve

z1 = 0.0
z2 = 60.0
zscale = 'nm'
Npts = 100

Grid1D = mesh.Mesh1D(z1,z2,zscale,Npts)

Efield = 1.25e6


V0 = 0
dV = V0 - cnvtr.convert_length(z2-z1)*Efield
V1 = V0 + dV

xmol = 0.15
mstar = 0.067*const.m_electron

wlayers = np.array([])

units = {'V':'V','layer':'nm','mstar':'kg'} 


V1D = mesh.Potential(Grid1D.z)
#V1D.linear_potential(Grid1D.z,V0,V1,wlayers,mstar,deltaV,units)

Solver = spsolve.SPsolve1D(Grid1D)