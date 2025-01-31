#include "timer.hh"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>

using namespace std;
using namespace orthrus;

constexpr double THOUSAND = 1e3;
constexpr double MILLION = 1e6;
constexpr double BILLION = 1e9;

template<class T>
class Value
{
private:
  T value;

public:
  Value( T v )
    : value( v )
  {
  }
  T get() const { return value; }
};

template<class T>
ostream& operator<<( ostream& o, const Value<T>& v )
{
  o << "\x1B[1m" << v.get() << "\x1B[0m";
  return o;
}

string Timer::summary() const
{
  size_t WIDTH = 25;

  for ( size_t i = 0; i < num_categories; i++ ) {
    WIDTH = max( WIDTH, strlen( _category_names.at( i ) ) );
  }

  WIDTH += 6;

  const uint64_t now = timestamp_ns();
  const uint64_t elapsed = now - _beginning_timestamp;

  ostringstream out;

  out << "Global timing summary:\n";

  out << "  " << left << setw( WIDTH - 2 ) << "Total time" << fixed << setprecision( 3 )
      << Value<double>( ( now - _beginning_timestamp ) / BILLION ) << " seconds\n";

  uint64_t accounted = 0;

  for ( unsigned int i = 0; i < num_categories; i++ ) {
    if ( _records.at( i ).count == 0 )
      continue;

    out << "    " << setw( WIDTH - 4 ) << left << _category_names.at( i );

    out << fixed << setprecision( 1 ) << Value<double>( 100 * _records.at( i ).total_ns / double( elapsed ) ) << "%";
    accounted += _records.at( i ).total_ns;

    uint64_t avg = _records.at( i ).total_ns / _records.at( i ).count;
    uint64_t var = _records.at( i ).total_sq_ns / _records.at( i ).count - avg * avg;
    uint64_t stdev = static_cast<uint64_t>( sqrt( var ) );

    out << "\x1B[2m [max=" << pp_ns( _records.at( i ).max_ns );
    out << ", avg=" << pp_ns( avg );
    out << ", min=" << pp_ns( _records.at( i ).min_ns );
    out << ", stdev=" << pp_ns( stdev );
    out << ", count=" << _records.at( i ).count << "]\x1B[0m";
    out << "\n";
  }

  const uint64_t unaccounted = elapsed - accounted;
  out << "    " << setw( WIDTH - 4 ) << "Unaccounted";
  out << fixed << setprecision( 1 ) << Value<double>( 100 * unaccounted / double( elapsed ) ) << "%\n";

  return out.str();
}

std::string Timer::pp_ns( const uint64_t duration_ns )
{
  ostringstream out;

  out << fixed << setprecision( 2 ) << setw( 5 );

  if ( duration_ns < THOUSAND ) {
    out << duration_ns << " ns";
  } else if ( duration_ns < MILLION ) {
    out << duration_ns / THOUSAND << " μs";
  } else if ( duration_ns < BILLION ) {
    out << duration_ns / MILLION << " ms";
  } else {
    out << duration_ns / BILLION << " s";
  }

  return out.str();
}
