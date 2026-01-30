#include "ndarray.hpp"
#include "pyarray_utils.hpp"

namespace gpaw::detail
{

void ArrayDesc::from_numpy(PyArrayObject* arr) noexcept
{
    data = PyArray_DATA(arr);
    ndim = static_cast<int32_t>(PyArray_NDIM(arr));
    shape = PyArray_DIMS(arr);
    strides = PyArray_STRIDES(arr);
    typenum = PyArray_TYPE(arr);
}

void ArrayDesc::from_cupy(PyObject* obj) noexcept
{
    // data.ptr
    PyObject* data_obj = PyObject_GetAttrString(obj, "data");
    if (data_obj)
    {
        PyObject* ptr_obj = PyObject_GetAttrString(data_obj, "ptr");
        if (ptr_obj)
        {
            data = PyLong_AsVoidPtr(ptr_obj);
            Py_DECREF(ptr_obj);
        }
        Py_DECREF(data_obj);
    }

    assert(data != nullptr && "Got a non-Cupy array?");

    // dtype
    typenum = gpaw::Array_TYPE(obj);

    // ndim
    PyObject* ndim_obj = PyObject_GetAttrString(obj, "ndim");
    if (ndim_obj)
    {
        ndim = static_cast<int32_t>(PyLong_AsLong(ndim_obj));
        Py_DECREF(ndim_obj);
    }

    // shape
    PyObject* shapeObj = PyObject_GetAttrString(obj, "shape");
    if (shapeObj)
    {
        shape = reinterpret_cast<int64_t*>(
            PyMem_Malloc(sizeof(int64_t) * ndim));
        for (int32_t i = 0; i < ndim; ++i)
        {
            shape[i] = PyLong_AsLongLong(
                PyTuple_GET_ITEM(shapeObj, i));
        }
        Py_DECREF(shapeObj);
    }

    // strides
    PyObject* stridesObj = PyObject_GetAttrString(obj, "strides");
    if (stridesObj)
    {
        strides = reinterpret_cast<int64_t*>(
            PyMem_Malloc(sizeof(int64_t) * ndim));
        for (int32_t i = 0; i < ndim; ++i)
        {
            strides[i] = PyLong_AsLongLong(
                PyTuple_GET_ITEM(stridesObj, i));
        }
        Py_DECREF(stridesObj);
    }
}

} // namespace gpaw::detail
