"""
This module has functionality to use non-linear least squares fitting to fit gaussian functions to the data points. (http://en.wikipedia.org/wiki/Non-linear_least_squares) This module is more efficient than scipy.opt.curvefit for gaussian fitting, because it solves a less general problem.
"""


import numpy
import math

#Gaussian function
def gaussian(x,A,mu,sigma):
  return A*math.exp(-(x-mu)**2/(2*sigma**2))

#Derivatives of gaussian w.r.t. listed variable
def g_A(x,A,mu,sigma):
  return math.exp(-(x-mu)**2/(2*sigma**2))

def g_mu(x,A,mu,sigma):
  return (A*(x-mu)/sigma**2)*math.exp(-(x-mu)**2/(2*sigma**2))

def g_sigma(x,A,mu,sigma):
  return (A*(x-mu)**2/sigma**3)*math.exp(-(x-mu)**2/(2*sigma**2))

#dB Builder
def dB_builder(x_list,y_list,A_guess,m_guess,s_guess):
  #Build dB vector
  dB = []
  for i in xrange(len(y_list)):
    dB.append(y_list[i] - gaussian(x_list[i],A_guess,m_guess,s_guess))
  dB = numpy.array(dB) 
  return dB


#Fitting function.
def fit_iteration(x_list,y_list,A_guess,m_guess,s_guess): 
  dB = dB_builder(x_list,y_list,A_guess,m_guess,s_guess)

  #Build matrix MdF
  MdF = []
  for i in xrange(len(x_list)):
    MdF.append([g_A(x_list[i],A_guess,m_guess,s_guess),g_mu(x_list[i],A_guess,m_guess,s_guess),g_sigma(x_list[i],A_guess,m_guess,s_guess)])
  MdF = numpy.array(MdF)

  #Set up normal equations
  MdF_trans = MdF.transpose()
  MtM = numpy.dot(MdF_trans,MdF)
  b = numpy.dot(MdF_trans,dB)
  
  delta_lambda = numpy.linalg.solve(MtM,b)
  return A_guess+delta_lambda[0],m_guess+delta_lambda[1],s_guess+delta_lambda[2]

#Function to call. Takes as args lists of x and y values, and optionally guesses for A,mu, and sigma, and number of iterations, returns improved guesses.
#Could easily be modified to repeat until coefficients are the same within required tolerance, but in practice this will cause outside programs to hang if initial guesses are too poor. Non-linear least squares relies on accurate initial estimates.
def gaussian_fit(x_list,y_list,A_guess=1.0,m_guess=0,s_guess=1.0,iterations=10):
  for q in xrange(0,iterations):
    A_guess,m_guess,s_guess = fit_iteration(x_list,y_list,A_guess,m_guess,s_guess)
  R = dB_builder(x_list,y_list,A_guess,m_guess,s_guess)
  return A_guess,m_guess,s_guess,(R[0]**2,R[1]**2,R[2]**2)
