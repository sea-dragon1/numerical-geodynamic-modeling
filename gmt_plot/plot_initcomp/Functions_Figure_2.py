from os import system as sys  # import all functions from os module
from numpy import *           # import numpy modules

# calctemp function
def calctemp(filename):
  
  # Define values
  k  = array([2.5,2.5,2.5])        # Thermal conductivity
  dz = array([40.e3,110.e3,200.e3]) # Layer thickness (m)
  A  = array([1.e-6,0.,0.])        # Radiogenic heat production (W/m**3)
  Tt = array([0.,550.,1500.])     # Temperature at top of layer
  Tb = array([550.,1500.,1360.])   # Temperature at base of layer
  qt = array([0.,0.,0.0])          # Heat flow at top of layer
  qb = array([0.0195,0.0,0.0])     # Heat flow at base of layer
  z  = array(range(0,241,1))*1.e3  # Depth from 0 to 240 km (1 km increments)
  t  = array(range(0,241,1))*0.    # Array for temperature values
  
  # Determine crustal top heat flow
  qt[0] = ( Tb[0] - Tt[0] + (A[0]*dz[0]**2)/(2.*k[0]) )*k[0]/dz[0]
  
  # Determine crustal basal (top lithospheric mantle) heat flow
  qb[0] = qt[0] - A[0]*dz[0]
  qt[1] = qb[0]
  
  # Determine lithospheric mantle thermal conductivity
  k[1]=(qt[1]*dz[1] - 0.5*A[1]*dz[1]**2)/(Tb[1]-Tt[1])
  
  # Determine lithosphere mantle basal (top sub-lithospheric mantle) heat flow
  qb[1] = qt[1] - A[1]*dz[1]
  qt[2] = qb[1]
  
  # Determine sub-lithospheric mantle thermal conductivity
  k[2]=(qt[2]*dz[2] - 0.5*A[2]*dz[2]**2)/(Tb[2]-Tt[2])
  
  # Determine sub-lithospheric mantle basal heat flow
  qb[2] = qt[2] - A[2]*dz[2]
  
  # Calculate temperature as a function of depth
  for i in range(size(z)):
    if z[i]<=dz[0]:
      t[i] = Tt[0] + (qt[0]/k[0])*z[i] - (A[0]*(z[i]**2))/(2*k[0])
    elif z[i]>=dz[0] and z[i]<=(dz[0]+dz[1]):
      t[i] = Tt[1] + (qt[1]/k[1])*(z[i]-dz[0]) - (A[1]*((z[i]-dz[0])**2))/(2*k[1])
    elif z[i]>=(dz[0]+dz[1]):
      t[i] = Tt[2] + (qt[2]/k[2])*(z[i]-dz[0]-dz[1]) - (A[2]*((z[i]-dz[0]-dz[1])**2))/(2*k[1])
  
  # Write temperature out to file
  if filename=='temp.dat':
    DataOut = column_stack((t,z/1.e3))
    savetxt(filename,DataOut,fmt=('%10.4f'))
  
  return t

# calcstrength function
def calcstrength(filename):
  
  # Define values
  z = array(range(0,241,1))*1.e3   # Depth from 0 to 240 km (1 km increments)
  d = array(range(0,241,1))*0.     # Density
  p = array(range(0,241,1))*0.     # Pressure
  v = array(range(0,241,1))*0.     # Viscosity related strength
  A = array(range(0,241,1))*0.     # Power-law constant
  Q = array(range(0,241,1))*0.     # Thermal activation energy
  n = array(range(0,241,1))*0.     # Power-law exponent
  b = array(range(0,241,1))*0.     # Brittle strength
  s = array(range(0,241,1))*0.     # Strength (min between brittle & viscous)
  t = calctemp('nowrite')          # Temperature
  c = 2.e7                         # Cohesion
  f1= 15.; f2 = 15.                # Friction angles (1 = initial, 2 = weakened)
  g = 9.8                          # Gravitational acceleration
  R = 8.31451                      # Gas law constant
  I2 = 1.e-14                      # Second invariant of strain-rate tensor
  
  
  # Assign depth-dependent values
  d[0:26]=2800.    ; d[26:36]=2900.    ; d[36:121]=3370.   ; d[121:241]=3370.
  A[0:26]=8.57e-30 ; A[26:36]=7.130e-18; A[36:121]=6.52e-16; A[121:241]=6.52e-16
  Q[0:26]=230.e3   ; Q[26:36]=345.e3   ; Q[36:121]=530.e3  ; Q[121:241]=530.e3
  n[0:26]=4.1       ; n[26:36]=3.       ; n[36:121]=3.5     ; n[121:241]=3.5
  
  # Calculate pressure
  for i in range(1,len(d)):
    p[i] = p[i-1] + d[i]*g*(z[i]-z[i-1])
  
  # Calculate brittle strength for initial friction angle value
  b[0:36] = p[0:36]*sin(f1*pi/180.) + c*cos(f1*pi/180.)
  b[36:241] = p[36:241]*sin(f2*pi/180.) + c*cos(f2*pi/180.)
  # Calculate effective strength related to viscosity
  for i in range(len(d)):
    v[i] = (A[i]**(-1./n[i])) * (I2**(1./(n[i]))) * exp(Q[i]/(n[i]*R*(t[i]+273.)))
  
  # Calculate minimum strength
  for i in range(len(d)): s[i] = min(b[i],v[i])
  
  # Write strength out to file
  DataOut = column_stack((s/1.e6,z/1.e3))
  savetxt('strength1.dat',DataOut,fmt=('%10.4f'))
  
  # Recalculate brittle strength for weakened friction angle value
  b[:] = p[:]*sin(f2*pi/180.) + c*cos(f2*pi/180.)
  
  # Write brittle strength (2) out to file
  DataOut = column_stack((b[0:51]/1.e6,z[0:51]/1.e3))
  savetxt('strength2.dat',DataOut,fmt=('%10.4f'))
  
  return v

if __name__ == "__main__":
    v = calcstrength('strength.dat')
