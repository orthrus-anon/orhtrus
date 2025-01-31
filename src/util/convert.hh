#pragma once

#include <cstdint>
#include <string_view>

namespace orthrus::util {

uint64_t to_uint64( const std::string_view str, const int base = 10 );

}
