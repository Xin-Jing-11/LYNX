// DeviceArray.cpp — explicit template instantiations for CPU builds.
// All logic lives in the header (DeviceArray.hpp); this file ensures
// that common instantiations are available in the lynx_core library.

#include "core/DeviceArray.hpp"
#include "core/types.hpp"
#include <complex>

namespace lynx {

template class DeviceArray<double>;
template class DeviceArray<float>;
template class DeviceArray<int>;
template class DeviceArray<Complex>;

}  // namespace lynx
