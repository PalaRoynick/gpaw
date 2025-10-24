/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "c_cpp_tools.h"
CLINKAGE typedef struct
{
  PyObject_HEAD
  int size;
  int rank;
  MPI_Comm comm;
  PyObject* parent;
  int* members;
} MPIObject;

