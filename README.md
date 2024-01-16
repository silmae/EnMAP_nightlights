# Analyzing artificial nighttime lighting using hyperspectral images from the EnMAP satellite

## Functionality
This repository contains methods for analyzing nighttime lighting using hyperspectral data from the German EnMAP satellite.
Basic functionaly of the code allows the used to:
 * load EnMAP level L1C images from ENVI files and the corresponding -xml metadata file of the datatake (works with the EnMAP folder structure or with just the three files in a folder)
 * plotting RGB composites from a spectral image
 * plotting spectra from a single pixel or from an area
   
These features are not limited to just nighttime imagery, though they may need adjustments. 

The code also contains functions to:
 * load library of laboratory-measured lighting spectra
 * use matched filtering with the lighting spectra as targets and plot the resulting maps
 * calculate and plot maps of the luminous efficiency of radiation and the spectral G index

## Dependencies
Install the EnPT Python package according to the instructions provided at: https://enmap.git-pages.gfz-potsdam.de/GFZ_Tools_EnMAP_BOX/EnPT/doc/installation.html

This package and its dependencies should contain everything you need to run the code in this repository. 

EnMAP data is freely available to registered users from the portal at: https://planning.enmap.org/

## Citing
If you find the code provided here useful, please cite this repository in anything you publish. An article related to the code presented here is in early stages of development, but a publication will happen sometime in the future.
