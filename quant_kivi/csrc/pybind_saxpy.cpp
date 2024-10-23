#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "saxpy.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("add", &add);
  m.def("main", &main);
}