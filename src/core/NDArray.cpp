// NDArray is header-only (template class).
// This file exists for the build system and future explicit instantiations.

#include "core/NDArray.hpp"

namespace lynx {

// Explicit instantiations for common types
template class NDArray<double>;
template class NDArray<float>;
template class NDArray<int>;
template class NDArray<std::complex<double>>;

} // namespace lynx
