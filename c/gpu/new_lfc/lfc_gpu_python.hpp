#pragma once
#include <pybind11/pybind11.h>
#include "new_lfc/lfc_builder.hpp"

/* Binds LFCGpu factory routines to the same Python-side class as the CPU version.
* Avoids needing to bind two separate classes that would in practice differ by just one function. */
void bind_lfc_gpu(pybind11::module_ m, pybind11::class_<LFCBuilder> cls);
