#include "atoms/AtomType.hpp"

namespace sparc {

AtomType::AtomType(const std::string& element, double mass, double Zval, int n_atoms)
    : element_(element), mass_(mass), Zval_(Zval), n_atoms_(n_atoms) {}

} // namespace sparc
