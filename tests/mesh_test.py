import grid.mesh as gmsh


#print(sys.path)

#print(hbar)
z1 = 0
z2 = 15 # [nm]
N = 100

length_scale = 'nm'
name = 'test mesh'

mesh = gmsh.Mesh1D(name)

mesh.generate(z1,z2,N,'nm')

mesh.inspect(length_scale='nm')

mesh.stats()