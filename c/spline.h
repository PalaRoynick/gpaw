#pragma once
/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "python_utils.h"
#include "extensions.h"
#include "bmgs/bmgs.h"

/* Need to expose this to new code on C++ side. Linkage won't work unless building everything as C++,
which is OK as the new code requires C++ mode anyway. */
extern PyTypeObject SplineType;

typedef struct
{
  PyObject_HEAD
  bmgsspline spline;
} SplineObject;
