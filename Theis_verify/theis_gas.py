########################################################################
#
# gas_inject.py
#
# Theis-like analaytical solution for (linearized) ideal gas injection
#
########################################################################

from numpy import *
from scipy.special import *
import matplotlib.pyplot as plt

########################################################################
#
# physical constants
#
########################################################################

g = 9.807                   # gravitational acceleration
R = 8.314                   # universal gas constant (J K-1 mol-1)

########################################################################
#
# classes
#
########################################################################

class Reservoir:
    def __init__(self,gas):
        # gas storage reservoir properties
        in_file = open('reservoir.txt','r')
        i = 0
        for line in in_file:
            line_input = line.split()
            if line_input[0] == 'permeability':
                self.k_inf = float(line_input[1])
            elif line_input[0] == 'klinkenberg_factor':
                self.b = float(line_input[1])
            elif line_input[0] == 'formation_thickness':
                self.h = float(line_input[1])                
            elif line_input[0] == 'porosity':
                self.phi = float(line_input[1])
            else:              # ambient gas pressure function
                self.Pb_i = float(line_input[1]) + self.b
            i += 1
        in_file.close()
        self.alpha = self.k_inf * self.Pb_i/(self.phi * gas.u)      # gas diffusivity
    def Theis(self,x,y,t,well,gas):
        # Theis solution for gas pressure perturbation at a location (x,y) with reference to line source
        r = sqrt((x - well.x)**2 + (y - well.y)**2)
        u = r**2.0/(4.0*self.alpha*t)
        dP2 = gas.u * well.Qm/(2.0*pi*self.k_inf*self.h*gas.beta) * -expi(-u)
        return dP2

class Gas:
    def __init__(self):
        # gas fluid properties
        in_file = open('gas.txt','r')
        i = 0
        for line in in_file:
            line_input = line.split()
            if line_input[0] == 'temperature':
                T = float(line_input[1])
            elif line_input[0] == 'viscosity':
                self.u = float(line_input[1])
            elif line_input[0] == 'molecular_weight':
                M = float(line_input[1])
            else:                   # injection rate (mass/time)
                self.Qm = float(line_input[1])
            i += 1
        in_file.close()
        self.beta = M/(R*T)    # compressibility factor

class WellVert:
    def __init__(self,name,x,y,Qm):
        # vertical well
        self.name = name
        self.x = x
        self.y = y
        self.Qm = Qm

########################################################################
#
# utility functions
#
########################################################################

# read vertical well(s) properties
def ReadWells():
    well_vert = []
    in_file = open('well_vert.txt','r')
    i = 0
    for line in in_file:
        if i > 0:                 # don't parse header
            line_input = line.split()
            name = line_input[0]
            x = float(line_input[1])
            y = float(line_input[2])
            Qm = float(line_input[3])
            well_vert.append(WellVert(name,x,y,Qm))
        i += 1
    in_file.close()
    return well_vert

# plot radial pressure profile leading away from well
def PlotRadialProfile(t,reservoir,well_vert_0,gas):
    r_max = float(raw_input('Maxium radial distance --> '))
    dr = r_max/100.0
    i = zeros(100) + range(1,101)
    x = i * dr
    y = 0
    Pb2 = reservoir.Theis(x,y,t,well_vert_0,gas)
    P = sqrt((reservoir.Pb_i-reservoir.b)**2 + Pb2)
    plt.plot(x,P/1.0e+5)                # plot pressure in bars
    plt.xlabel('Distance (m)')
    plt.ylabel('Pressure (bar)')
    plt.show()
    # write pressures to output file for final time step
    out_file = open('theis_gas.txt','w')
    line_out = []
    line_out.append('Radial_dist.')
    line_out.append('\t')
    line_out.append('Pressure')
    line_out.append('\n')
    out_file.writelines(line_out)
    for i in xrange(len(x)):
        line_out = []
        line_out.append(str(x[i]))
        line_out.append('\t')
        line_out.append(str(P[i]))
        line_out.append('\n')
        out_file.writelines(line_out)
    out_file.close()


########################################################################
#
# main script
#
########################################################################

def GasFlow():

    t = float(raw_input('monitor time --> '))    

    gas = Gas()
    print 'Read gas data.'

    reservoir = Reservoir(gas)
    print 'Read reservoir data.'

    well_vert = ReadWells()
    print 'read data for vertical well(s).'

    # calculate and plot radial profile for example well; write output to file
    PlotRadialProfile(t,reservoir,well_vert[0],gas)

    print 'Done.'


###################################

GasFlow()
