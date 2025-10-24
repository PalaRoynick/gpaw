#pragma once

#include "c_cpp_tools.h"
#include "extensions.h"

/* Ensure C-linkage for ELPA declarations. Modern ELPA has these inside the header already,
but seems to be required for compatibility with older versions. */
CLINKAGE_BEGIN
#include <elpa/elpa.h>
CLINKAGE_END
