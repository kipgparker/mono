#include "cache.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reshape_and_cache_flash", &reshape_and_cache_flash);
}