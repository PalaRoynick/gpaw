#pragma once

#include "pyarray_utils.hpp"

namespace gpaw
{

/* Aggressive CPython-style accessors to array attributes.
* For Numpy these are trivial wrappers around Numpy C-API.
*/
struct ArrayAccessors
{
    // ===================== NumPy =====================

    static void* data(PyArrayObject* obj) noexcept
    {
        return PyArray_DATA(obj);
    }

    static int typenum(PyArrayObject* obj) noexcept
    {
        return PyArray_TYPE(obj);
    }

    static bool is_c_contig(PyArrayObject* obj) noexcept
    {
        return PyArray_IS_C_CONTIGUOUS(obj);
    }

    static bool is_f_contig(PyArrayObject* obj) noexcept
    {
        return PyArray_IS_F_CONTIGUOUS(obj);
    }

    static int32_t ndim(PyArrayObject* obj) noexcept
    {
        return static_cast<int32_t>(PyArray_NDIM(obj));
    }

    static std::vector<int64_t> shape(PyArrayObject* obj) noexcept
    {
        const int32_t num_dims = ndim(obj);
        std::vector<int64_t> out(num_dims);
        npy_intp* shape_ptr = PyArray_DIMS(obj);
        for (int32_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)shape_ptr[i];
        }
        return out;
    }

    static std::vector<int64_t> strides(PyArrayObject* obj) noexcept
    {
        const int32_t num_dims = ndim(obj);
        std::vector<int64_t> out(num_dims);
        npy_intp* stride_ptr = PyArray_STRIDES(obj);
        for (int32_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)stride_ptr[i];
        }
        return out;
    }

    static void fill_shape(PyArrayObject* obj, int64_t* out, int32_t n) noexcept
    {
        npy_intp* dims = PyArray_DIMS(obj);
        for (int32_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<int64_t>(dims[i]);
        }
    }

    static void fill_strides(PyArrayObject* obj, int64_t* out, int32_t n) noexcept
    {
        npy_intp* strides = PyArray_STRIDES(obj);
        for (int32_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<int64_t>(strides[i]);
        }
    }

    // ===================== CuPy =====================

    static void* data(PyObject* obj) noexcept
    {
        // obj.data.ptr (device pointer)
        PyObject* data_obj = PyObject_GetAttrString(obj, "data");
        PyObject* data_ptr_obj = data_obj ? PyObject_GetAttrString(data_obj, "ptr") : nullptr;
        void* ptr = data_ptr_obj ? PyLong_AsVoidPtr(data_ptr_obj) : nullptr;
        Py_XDECREF(data_ptr_obj);
        Py_XDECREF(data_obj);
        return ptr;
    }

    static int typenum(PyObject* obj) noexcept
    {
        // obj.dtype.num
        PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
        PyObject* typenum_obj = dtype ? PyObject_GetAttrString(dtype, "num") : nullptr;
        const int typenum = typenum_obj ? static_cast<int>(PyLong_AsLong(typenum_obj)) : -1;
        Py_XDECREF(typenum_obj);
        Py_XDECREF(dtype);
        return typenum;
    }

    static bool is_c_contig(PyObject* obj) noexcept
    {
        // obj.flags.c_contiguous
        PyObject* flags = PyObject_GetAttrString(obj, "flags");
        PyObject* c_contig_obj = flags ? PyObject_GetAttrString(flags, "c_contiguous") : nullptr;
        const bool value = c_contig_obj ? PyObject_IsTrue(c_contig_obj) : false;
        Py_XDECREF(c_contig_obj);
        Py_XDECREF(flags);
        return value;
    }

    static bool is_f_contig(PyObject* obj) noexcept
    {
        // obj.flags.f_contiguous
        PyObject* flags = PyObject_GetAttrString(obj, "flags");
        PyObject* f_contig_obj = flags ? PyObject_GetAttrString(flags, "f_contiguous") : nullptr;
        const bool value = f_contig_obj ? PyObject_IsTrue(f_contig_obj) : false;
        Py_XDECREF(f_contig_obj);
        Py_XDECREF(flags);
        return value;
    }

    static int32_t ndim(PyObject* obj) noexcept
    {
        // obj.ndim
        PyObject* ndim_obj = PyObject_GetAttrString(obj, "ndim");
        const int32_t ndim = ndim_obj ? static_cast<int32_t>(PyLong_AsLong(ndim_obj)) : -1;
        Py_XDECREF(ndim_obj);
        return ndim;
    }

    static void fill_shape(PyObject* obj, int64_t* out, int32_t n) noexcept
    {
        PyObject* shape_tuple = PyObject_GetAttrString(obj, "shape");
        if (!shape_tuple)
        {
            return;
        }

        for (int32_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<int64_t>(PyLong_AsLongLong(PyTuple_GET_ITEM(shape_tuple, i)));
        }
        Py_DECREF(shape_tuple);
    }

    static void fill_strides(PyObject* obj, int64_t* out, int32_t n) noexcept
    {
        PyObject* strides_tuple = PyObject_GetAttrString(obj, "strides");
        if (!strides_tuple)
        {
            return;
        }

        for (int32_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<int64_t>(PyLong_AsLongLong(PyTuple_GET_ITEM(strides_tuple, i)));
        }
        Py_DECREF(strides_tuple);
    }

    static std::vector<int64_t> shape(PyObject* obj) noexcept
    {
        // obj.shape (tuple)
        PyObject* shape_tuple = PyObject_GetAttrString(obj, "shape");
        if (!shape_tuple)
        {
            return {};
        }

        const Py_ssize_t num_dims = PyTuple_GET_SIZE(shape_tuple);

        std::vector<int64_t> out(num_dims);
        for (Py_ssize_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)PyLong_AsLongLong(PyTuple_GET_ITEM(shape_tuple, i));
        }

        Py_DECREF(shape_tuple);
        return out;
    }

    static std::vector<int64_t> strides(PyObject* obj) noexcept
    {
        // obj.strides (tuple)
        PyObject* strides_tuple = PyObject_GetAttrString(obj, "strides");
        if (!strides_tuple)
        {
            return {};
        }

        const Py_ssize_t num_dims = PyTuple_GET_SIZE(strides_tuple);

        std::vector<int64_t> out(num_dims);
        for (Py_ssize_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)PyLong_AsLongLong(PyTuple_GET_ITEM(strides_tuple, i));
        }

        Py_DECREF(strides_tuple);
        return out;
    }

};

} // namespace gpaw
