#pragma once

#ifdef __cplusplus
    #define CLINKAGE

    /* GPAW has lots of C code that needs C-linkage (all Python bindings)
    so the following are handy. Note that blocks like CLINKAGE { (some defs) }
    are not valid if compiling as C code. */

    // Starts an extern "C" block
    #define CLINKAGE_BEGIN
    // Ends an extern "C" block
    #define CLINKAGE_END
    #define ALWAYSCLINKAGE extern "C"
    #define ALWAYSCLINKAGE_BEGIN extern "C" {
    #define ALWAYSCLINKAGE_END }
#else
    #define CLINKAGE
    #define CLINKAGE_BEGIN
    #define CLINKAGE_END
    #define ALWAYSCLINKAGE
    #define ALWAYSCLINKAGE_BEGIN
    #define ALWAYSCLINKAGE_END
#endif


/* Sets a runtime error to the Python interpreter. Caller should check with PyErr_Occured()
and proceed accordingly. */
CLINKAGE void gpaw_set_runtime_error(const char* err_msg);
// Aborts the program as gracefully as possible. Also raises a Python error
CLINKAGE void gpaw_abort(const char* err_msg);
