#include <filesystem>
#include <iostream>
#include <vector>

#include <glog/logging.h>

#define OOF_IMPL
#include "oof/oof.hh"

#include "compute/kernel_hybrid_simple.hh"
#include "models/common/state.hh"
#include "models/llama2/model.hh"
#include "util/timer.hh"

#include "arch/platform_macros.hh"

using namespace std;
using namespace orthrus;
using namespace orthrus::models;

template<typename ModelA, typename ModelB>
class BatchInference
{
private:
  using StateType = BatchedInferenceState<typename ModelA::ConfigType>;

  const uint32_t batch_size_;
  const float temp_;

  // TODO: fix this
  compute::SimpleHybridComputeKernel<ModelA, ModelB> kernel_;
  llama2::Vocabulary vocabulary_;
  StateType state_;

  std::vector<uint32_t> tokens_ { 1 /* BOS */ };

  size_t auto_id_ { 0 };

  HashID next_hash_id()
  {
    HashID id;
    util::digest::sha256( to_string( auto_id_++ ), id );
    return id;
  }

  StateType ser_des( StateType&& state )
  {
    const std::string ser = state.serialize();
    state = {};

    return StateType { ser };
  }

  StateType make_state()
  {
    StateType st { batch_size_, DataType::_ORTHRUS_DTYPE_NAME_, {}, {} };

    // TODO: have to be careful with how many context ids we are making
    for ( size_t i = 0; i < batch_size_; ++i ) {
      st.set_prompt( i, next_hash_id(), NULL_CONTEXT, 1 /* BOS */, 0, temp_, 1, 0, 0, 0 );
    }

    st.set_next_layer( 0 );
    st.set_next_stage( InferenceStage::PreAttention );

    return st;
  }

public:
  BatchInference( const filesystem::path& model_path,
                  const filesystem::path& tokenizer_path,
                  const size_t batch_size,
                  const float temp )
    : batch_size_( batch_size )
    , temp_( temp )
    , kernel_( [&]() -> decltype( kernel_ ) {
      std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>,
                 ModelA::ConfigType::n_layers>
        hosting_table;
      for ( size_t i = 0; i < ModelA::ConfigType::n_layers; i++ ) {
        for ( size_t j = 0; j < util::to_underlying( models::InferenceStage::__COUNT__ ); j++ ) {
          if ( i < ModelA::ConfigType::n_layers - 1
               and j == util::to_underlying( models::InferenceStage::Classification ) ) {
            hosting_table[i][j] = false;
          } else {
            hosting_table[i][j] = true;
          }
        }
      }
      return { batch_size, model_path, hosting_table, batch_size, batch_size };
    }() )
    , vocabulary_( tokenizer_path )
    , state_( make_state() )
  {
    for ( size_t i = 0; i < batch_size_; ++i ) {
      state_.set_context_id( i, NULL_CONTEXT + i );
    }
  }

  void run()
  {
    static const oof::color colors[] = {
      oof::color { 255, 0, 0 },     oof::color { 0, 255, 0 },     oof::color { 0, 0, 255 },
      oof::color { 255, 255, 0 },   oof::color { 255, 0, 255 },   oof::color { 0, 255, 255 },
      oof::color { 128, 0, 0 },     oof::color { 0, 128, 0 },     oof::color { 0, 0, 128 },
      oof::color { 128, 128, 0 },   oof::color { 128, 0, 128 },   oof::color { 0, 128, 128 },
      oof::color { 128, 128, 128 }, oof::color { 192, 0, 0 },     oof::color { 0, 192, 0 },
      oof::color { 0, 0, 192 },     oof::color { 192, 192, 0 },   oof::color { 192, 0, 192 },
      oof::color { 0, 192, 192 },   oof::color { 192, 192, 192 },
    };

    std::random_device rd;
    std::mt19937 gen { rd() };
    std::uniform_real_distribution<float> dis { 0.0, 1.0 };

    kernel_.event_fd().set_blocking( true );

    for ( size_t pos = 0; pos < ModelA::ConfigType::seq_len; pos++ ) {
      kernel_.push( std::move( state_ ) );
      kernel_.event_fd().read_event(); // blocks until results are ready
      kernel_.pop( state_ );

      DCHECK( kernel_.pop( state_ ) == false );

      if ( temp_ == 0 ) {
        // random chance to terminate a prompt early (otherwise they will all have the same length)
        for ( size_t i = 0; i < state_.batch_size(); ++i ) {
          if ( state_.token_pos( i ) >= 128 and dis( gen ) < 0.05 ) {
            state_.set_finished( i );
          }
        }

        bool any_finished = false;
        for ( size_t i = 0; i < state_.batch_size(); ++i ) {
          if ( state_.finished( i ) ) {
            state_.discard( i );
            any_finished = true;
            cout << oof::fg_color( i == 0 ? oof::color { 255, 255, 255 } : colors[( i - 1 ) % sizeof( colors )] )
                 << "<|" << oof::reset_formatting();
          }
        }

        if ( any_finished ) {
          auto new_state = make_state();
          state_.replenish_from( new_state );
          DCHECK_EQ( state_.free_slots(), 0 );
        }
      }

      const auto token = state_.token( 0 );
      cout << vocabulary_.get_word( token ) << flush;

      for ( size_t i = 1; i < state_.batch_size(); i++ ) {
        if ( token != state_.token( i ) ) {
          cout << oof::fg_color( colors[( i - 1 ) % sizeof( colors )] ) << "["
               << vocabulary_.get_word( state_.token( i ) ) << "]" << oof::reset_formatting();
        }
      }
    }
  }
};

void usage( const char* argv0 )
{
  cerr << "Usage: " << argv0 << " <model_dir> <model_name> <tokenizer_path> <batch_size> <temperature>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 6 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir_path { argv[1] };
    const string model_name { argv[2] };
    const filesystem::path tokenizer_path { argv[3] };
    const size_t batch_size = atoi( argv[4] );
    const float temp = atof( argv[5] );

    using ModelTypeA = llama2::cuda::Stories_110M_Static<orthrus::float16_t>;
    using ModelTypeB = llama2::amd64::Stories_110M_Static<orthrus::float32_t>;

    BatchInference<ModelTypeA, ModelTypeB> inference { model_dir_path, tokenizer_path, batch_size, temp };
    inference.run();

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
