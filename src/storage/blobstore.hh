#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace orthrus::storage {

enum class OpResult
{
  OK,
  NotFound,
  Error,
  InvalidKey,
};

class BlobStore
{
public:
  static std::unique_ptr<BlobStore> create( const std::string& uri );

public:
  BlobStore() {};
  virtual ~BlobStore() {}

  virtual std::pair<OpResult, std::string> get( const std::string& key ) = 0;
  virtual OpResult put( const std::string& key, const std::string& value ) = 0;
  virtual OpResult remove( const std::string& key ) = 0;

  virtual std::vector<std::pair<OpResult, std::string>> get( const std::vector<std::string>& keys ) = 0;
  virtual std::vector<OpResult> put( const std::vector<std::pair<std::string, std::string>>& kvs ) = 0;
  virtual std::vector<OpResult> remove( const std::vector<std::string>& keys ) = 0;

  virtual std::string to_string() const = 0;
};

} // namespace orthrus::storage
