########################################################################################################################################
#
# wg_1D.py
#
# * a 1-D fully-coupled flow model for gas and water phases in arbitrarily-shaped flow tube
# * pressure and water saturation PDEs are solved using numerical method of lines
# * influence of pressure on water saturation is quantified through the capillary pressure equation
# * influence of water saturation on pressure is quantified through available void volume
#
# assumptions:
#
# (1) system is isothermal
# (2) density-driven flow for gas phase is ignored
#
########################################################################################################################################

from numpy import *
from scipy.integrate import quad
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# constants ...
g = 9.807               # gravitational acceleration (m/sec^2)
R = 8.314               # universal gas constant (J K-1 mol-1)
MW_g = 0.029250065      # gas molecular weight
rho_w = 1000.           # density of water (kg/m^3)
u_w = 8.9e-4            # viscosity of water (Pa*sec)
u_g = 1.84e-5           # gas viscosity

###################################
#
# classes
#
###################################

class Grid:
    def __init__(self,material,material_locs):
        # define geometry
        x0,xf,self.N,theta,f = self.ReadGeometry()
        self.dx = (xf-x0)/self.N                                                # grid spacing
        x_u = arange(x0,xf,self.dx)                                             # upstream boundary cross-sectional area
        x_d = arange(x0,xf,self.dx)+self.dx                                     # downstream boundary cross-sectional area
        self.x = 0.5*x_u + 0.5*x_d                                              # node location along column
        self.grad_z = cos(theta)                                            # elevation associated with node location x
        A = lambda x: eval(f)                                              # self.A(x) = cross-sectional area as a function of x, where f = 'x**2 + 4' or some similar expression
        self.A_u = A(x_u)                                                  # upstream boundary cross-sectional area
        self.A_d = A(x_d)                                                  # downstream boundary cross-sectional area
        self.vol = zeros(self.N,float)                                          # compute bulk volumes of individual grids
        for i in xrange(self.N): self.vol[i] = quad(A,x_u[i],x_d[i])[0]
        # assign materials
        material_indexes = self.MatAssignment(material_locs)                    # match material index numbers to grids
        self.name = []                                                          # name of material/soil/rock type
        self.k0 = zeros(self.N,float)                                           # intrinsic permeability
        self.phi = zeros(self.N,float)                                          # porosity
        self.alpha = zeros(self.N,float)                                        # Van Genuchten unsaturated flow model parameters
        self.n = zeros(self.N,float)
        self.m = zeros(self.N,float)
        self.Sr = zeros(self.N,float)                                           # residual saturation of water
        for i in xrange(self.N):
            self.name.append(material[material_indexes[i]][0])
            self.k0[i] = material[material_indexes[i]][1]
            self.phi[i] = material[material_indexes[i]][2]                                        
            self.alpha[i] = material[material_indexes[i]][3]                                    
            self.n[i] = material[material_indexes[i]][4]
            self.m[i] = 1.0 - 1.0/material[material_indexes[i]][4]
            self.Sr[i] = material[material_indexes[i]][5]                                          
    def PermEff(self,S):
        # return relative permeability for both water and gas (note that this function is vectorized across all cells)
        Se = (S - self.Sr)/(1. - self.Sr)
        k_w = Se**0.5 * (1.0 - (1.0 - Se**(1.0/self.m))**self.m)**2
        k_g = (1.-Se)**0.5 * (1.-Se**(1.0/self.m))**(2.*self.m)
        return k_w,k_g
    def Q_g(self,P,S,T):
        # return a vector of gas fluxes across all cell connections (use upstream weighting)
        grad = (P[:-1]-P[1:])/self.dx
        upstream_K = ((0.5+0.5*sign(grad))*self.PermEff(S)[1][:-1]*self.k0[:-1] + (0.5-0.5*sign(grad))*self.PermEff(S)[1][1:]*self.k0[1:]) / u_g
        flux = MW_g*(0.5*P[:-1] + 0.5*P[1:])/(R*T) * self.A_d[:-1] * upstream_K * grad          # mass flux of gas across connection (mass/time)
        flux = append(array([0.0]),flux)                # append 0's to ends of flux array (to facilitate processing by ODEs function)
        flux = append(flux,array([0.0]))
        return flux
    def Q_w(self,P,S):
        # return a vector of water fluxes across all cell connections (use upstream weighting)
        Pc = -self.Pcap(S)
        grad = ((P[:-1]-Pc[:-1]) - (P[1:]-Pc[1:]))/self.dx + rho_w*g*self.grad_z
        upstream_K = ((0.5+0.5*sign(grad))*self.PermEff(S)[0][:-1]*self.k0[:-1] + (0.5-0.5*sign(grad))*self.PermEff(S)[0][1:]*self.k0[1:]) / u_w
        flux = self.A_d[:-1] * upstream_K * grad                    # volumetric flux of water across connection (vol/time)
        flux = append(array([0.0]),flux)                # append 0's to ends of flux array (to faciliate processing by CoupledODES method)
        flux = append(flux,array([0.0]))
        return flux
    def Pcap(self,S):
        # return capillary pressure as a function of water saturation (note that this function is vectorized across all cells)
        Se = (S - self.Sr)/(1. - self.Sr)
        return -exp(log(exp(-log(Se)/self.m) - 1.0)/self.n-log(self.alpha))
    def ReadGeometry(self):
        input_file = open('grid.txt','r')
        line_input = []
        for line in input_file: line_input.append(line.split())
        input_file.close()
        x0 = float(line_input[0][1])                                        # starting point
        xf = float(line_input[1][1])                                        # ending point
        N = int(line_input[2][1])                                           # number of divisions along the 1-D section
        theta = float(line_input[3][1]) * pi/180.                           # deviation of 1-D column from vertical
        f = line_input[4][1]                                                # equation used to express A = f(x)
        return x0,xf,N,theta,f
    def MatAssignment(self,material_locs):
        # return an array of index numbers corresponding to material assignments
        mat_indexes = zeros(self.N,int)         # note: default material is the first material in the materials objects list
        for i in xrange(len(material_locs)): mat_indexes += material_locs[i][1] * (self.x > material_locs[i][2]).astype(int) * (self.x <= material_locs[i][3]).astype(int)
        return mat_indexes

class Source:
    def __init__(self,grid):
        input_file = open('sources.txt','r')
        line_input = []
        i = 0
        for line in input_file:
            if i: line_input.append(line.split())           # don't parse header
            i += 1
        input_file.close()
        self.Q_g = zeros(grid.N,float)                      # gas fluxes
        self.Q_w = zeros(grid.N,float)                      # water fluxes
        self.fixed = zeros(grid.N,float)                    # constant potential cells
        for j in xrange(len(line_input)):
            i_cell = int(line_input[j][0])
            self.Q_g[i_cell] += float(line_input[j][2])
            self.Q_w[i_cell] += float(line_input[j][3])
            if int(line_input[j][1]): self.fixed[i_cell] += 1e+30
    def FixCells(self,grid):
        # specify fixed potential cells with large volumes
        grid.vol += self.fixed
        return grid


###################################
#
# support functions
#
###################################


def ReadMaterials():
    material = []
    material_list = []
    input_file = open('materials.txt','r')
    line_input = []
    for line in input_file: line_input.append(line.split())
    input_file.close()
    for i in xrange(len(line_input)):
        if i:
            material.append([line_input[i][0],float(line_input[i][1]),float(line_input[i][2]),float(line_input[i][3]),float(line_input[i][4]),float(line_input[i][5])])
            material_list.append(line_input[i][0])
    return material,material_list

def ReadMatList(material_list):
    # read in material position assignments (name, starting point, ending point)
    mat_list = []
    input_file = open('mat_list.txt','r')
    line_input = []
    for line in input_file: line_input.append(line.split())
    input_file.close()
    for i in xrange(len(line_input)):
        if i:
            index_num = material_list.index(line_input[i][0])
            mat_list.append([line_input[i][0],index_num,float(line_input[i][1]),float(line_input[i][2])])
    return mat_list

def ReadInitConds(grid,P0,S0):
    # read in initial conditions for water and gas phases
    input_file = open('ICs.txt','r')    
    line_input = []
    i = 0
    for line in input_file:
        if i: line_input.append(line.split())           # don't parse header
        i += 1
    input_file.close()
    P = zeros(grid.N,float) + P0
    S = zeros(grid.N,float) + S0
    for j in xrange(len(line_input)):
        i_cell = int(line_input[j][0])
        P[i_cell] = float(line_input[j][1])
        S[i_cell] = float(line_input[j][2])
    return P,S

def ODEsys(y,t,grid,source,T):
    # solve coupled ODEs for pressure and water saturation
    dy = zeros(2*grid.N,float)        
    # pressure equations
    dy[:grid.N] = R*T/(MW_g*grid.vol*grid.phi*(1.-y[grid.N:])) \
        * (source.Q_g + grid.Q_g(y[:grid.N],y[grid.N:],T)[:-1] - grid.Q_g(y[:grid.N],y[grid.N:],T)[1:])
    # saturation equations
    dy[grid.N:] = 1.0/(grid.vol*grid.phi) \
        * (source.Q_w + grid.Q_w(y[:grid.N],y[grid.N:])[:-1] - grid.Q_w(y[:grid.N],y[grid.N:])[1:])
    return dy

def WriteOutput(i_step,y,t,grid,prefix,i_finish):
    # write header
    output_file = open(prefix + '.txt','w')
    line_out = ['x','\t','P','\t','S','\n']
    output_file.writelines(line_out)
    # write model results for time t[i_step]
    t_slice = y[i_step]
    P = t_slice[:grid.N]
    S = t_slice[grid.N:]
    for i in xrange(grid.N):
        line_out = [str(grid.x[i]),'\t',str(P[i]),'\t',str(S[i]),'\n']
        output_file.writelines(line_out)
    output_file.close()
    # display saturation results on graph at end-of-run
    if i_finish:
        plt.plot(grid.x,S)
        plt.xlabel('Position')
        plt.ylabel('Saturation')
        plt.show()

def ReadTimes():
    # read in time steps for which to write output
    t = []
    i = 0
    times_file = open('times.txt','r')
    for line in times_file:
        line_input = line.split()
        if i:t.append(float(line_input[0]))
        i += 1
    times_file.close()
    return array(t)



###################################
#
# script
#
###################################

def multiflow(T,P0,S0):

    # process model definition

    material,material_list = ReadMaterials()
    print 'Processed material properties.'

    material_locs = ReadMatList(material_list)
    print 'Processed material location assignments.'

    grid = Grid(material,material_locs)
    print 'Set up grid geometry and distributed properties.'

    source = Source(grid)
    grid = source.FixCells(grid)         # assign fixed-potential cells, as applicable    
    print 'Processed source term(s)/boundary conditions.'

    P,S = ReadInitConds(grid,P0,S0)
    print 'Processed initial conditions.'

    t = ReadTimes()
    print 'Read output times.'

    # solve PDE by method-of-lines
    print ' '
    print 'Solving PDE ...'
    y0 = concatenate((P,S),axis=0)    
    y = odeint(ODEsys,y0,t,mxstep=5000,args=(grid,source,T))   

    # output - publish pressure and saturation and plot final saturation state
    i_finish = 0
    for i_step in xrange(len(t)):
        prefix = 'out_' + str(t[i_step])
        if i_step == len(t)-1:i_finish = 1
        WriteOutput(i_step,y,t,grid,prefix,i_finish)

    print 'Done.'


##### run script #####

multiflow(298.15,1.013e+5,0.5) #note: arguments = default temperature, default pressure, and default water saturation 
    

