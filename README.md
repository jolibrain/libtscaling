# libtscaling
Temperature scaling for neural network calibration

compile against liblbfgs-dev on ubuntu

 g++   -O2 tscaling.cpp -std=c++11 -o cal -llbfgs


compile using eigen against LBFGSPP (included):
g++  -DLBFGSPP -I/usr/include/eigen3  -O2 tscaling.cpp -std=c++11 -o cal 
