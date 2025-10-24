#pragma once

/* This header collects declarations of Python-bound functions that were
previously in _gpaw.h. Having them in a dedicated header is useful because these
all need C-linkage. So we wrap the declarations in a C-linkage block here and
simply #include this header in files that provide definitions.
*/

#include "c_cpp_tools.h"

