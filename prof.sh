#!/bin/bash

clear
nsys nvprof ./lab1
nsys stats --report gpusum report1.nsys-rep
