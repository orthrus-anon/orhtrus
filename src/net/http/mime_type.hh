#pragma once

#include <string>
#include <vector>

namespace orthrus::net {

class MIMEType
{
private:
  std::string type_;
  std::vector<std::pair<std::string, std::string>> parameters_;

public:
  MIMEType( const std::string_view content_type );

  const std::string& type() const { return type_; }
};

} // namespace orthrus::net
