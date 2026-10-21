/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "python_utils.h"
#ifdef PARALLEL
  #include <mpi.h>
#endif

typedef struct
{
  PyObject_HEAD
  int size;
  int rank;
  MPI_Comm comm;
  PyObject* parent;
  int* members;
} MPIObject;

