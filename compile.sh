#!/bin/bash


#module load anaconda3-2024.10-1

mkdir -p lib fmod

cd ftn
gfortran -L$CONDA_PREFIX/lib  -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -shared f_numc.f90 -o ../lib/libfnumc.so -J../fmod -fPIC
cd ..

cd ftn
gfortran -L$CONDA_PREFIX/lib  -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -L../lib -lfnumc -shared f_tools.f90 -o ../lib/libftools.so -J../fmod -fPIC
cd ..

cd ftn
gfortran -L$CONDA_PREFIX/lib  -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -L../lib -lfnumc -lftools -shared f_tpw.f90 -o ../lib/libftpw.so -J../fmod -fPIC
cd ..


#[Optional]  gfortran test.f90 -J./fmod -L./lib -lfedvr -ltools -lspf -o test

export LD_LIBRARY_PATH=/home/imam77/Programs/tPW/lib:$LD_LIBRARY_PATH

