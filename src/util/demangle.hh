#pragma once

#include <string>

namespace orthrus::util {

std::string demangle( const std::string& name, const bool keep_template_args = true );

} // namespace orthrus::util
