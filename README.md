# Wildfire 0D Model
Python implementation of an ADR wildfire model for temperature and fuel-fraction evolution in a spatially homogeneous setting.

This repository contains the code used to simulate a spatially homogeneous wildfire model derived from an advection-diffusion-reaction formulation. In this zero-dimensional version, spatial dependence is neglected and the temporal evolution of temperature and remaining fuel fraction is analyzed.

# Model description
The model describes the evolution of two variables:
  T: temperature [K]
  Y: remaining fuel fraction [-]

The temperature equation includes the contribution of:
  a reaction term,
  a cooling term,
  a volatilization term, depending on the selected model configuration.
The fuel fraction evolves according to a temperature-dependent reaction rate.

# Model configuration
The script allows different model configurations through the following parameters:
  VOLATILIZATION_PRODUCTS = 0   # 0: simplified model, 1: complete model
  COOLING_MODEL = 0             # 0: convection, 1: convection + radiation
The available configurations are:
  Simplified:	constant air density and convective cooling
  Simplified with radiation: constant air density and convective-radiative cooling
  Complete:	temperature-dependent air density and volatilization term
  Complete with radiation:	temperature-dependent air density, volatilization term and radiation

# Output figures

Depending on the selected model configuration, the script generates figures such as:
  figures/ContribucionTerminos_Simplificado.png
  figures/EvolucionTemporal_Simplificado.png
  figures/Trayectoria_EspacioFases_Simplificado.png
  figures/EspacioFases_Simplificado.png

The figures include:
  contribution of the thermal terms,
  temporal evolution of temperature and fuel fraction,
  individual trajectory in phase space,
  phase-space vector field.
