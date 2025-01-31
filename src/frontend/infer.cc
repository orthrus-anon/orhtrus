#include <filesystem>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>

#include "models/common/state.hh"
#include "models/llama2/model.hh"
#include "prompt/prompt.hh"
#include "util/timer.hh"

#include "arch/platform_macros.hh"

#define OOF_IMPL
#include "oof/oof.hh"

using namespace std;
using namespace orthrus;
using namespace orthrus::models;

template<class Model>
class BatchInference
{
private:
  using StateType = BatchedInferenceState<typename Model::ConfigType>;

  const uint32_t batch_size_;
  const float temp_;

  Model model_;
  llama2::Vocabulary vocabulary_;
  StateType state_;

  queue<prompt::Prompt> prompt_queue_ {};
  vector<prompt::Prompt> active_prompts_ {};
  vector<typename Model::ContextPtr> active_contexts_ {};
  vector<prompt::Prompt> finished_prompts_ {};

  ofstream completions_file_;

public:
  BatchInference( const filesystem::path& model_path,
                  const filesystem::path& tokenizer_path,
                  const filesystem::path& prompts_path,
                  const filesystem::path& completions_path,
                  const size_t batch_size,
                  const float temp )
    : batch_size_( batch_size )
    , temp_( temp )
    , model_( [&]() -> Model {
      std::array<std::array<bool, util::to_underlying( models::InferenceStage::__COUNT__ )>,
                 Model::ConfigType::n_layers>
        hosting_table;
      for ( size_t i = 0; i < Model::ConfigType::n_layers; i++ ) {
        for ( size_t j = 0; j < util::to_underlying( models::InferenceStage::__COUNT__ ); j++ ) {
          if ( i < Model::ConfigType::n_layers - 1
               and j == util::to_underlying( models::InferenceStage::Classification ) ) {
            hosting_table[i][j] = false;
          } else {
            hosting_table[i][j] = true;
          }
        }
      }
      return { model_path, hosting_table, batch_size, batch_size };
    }() )
    , vocabulary_( tokenizer_path )
    , state_( batch_size, DataType::_ORTHRUS_DTYPE_NAME_, {}, {} )
    , completions_file_( completions_path, ios::out | ios::trunc )
  {
    ifstream prompts_file { prompts_path }; // JSONL file of prompts
    CHECK( prompts_file.is_open() ) << "Failed to open prompts file: " << prompts_path;

    string line;
    while ( getline( prompts_file, line ) ) {
      prompt_queue_.push( prompt::Prompt::from_json( line ) );
    }

    LOG( INFO ) << "Loaded " << prompt_queue_.size() << " prompts.";

    state_.set_next_layer( 0 );
    state_.set_next_stage( InferenceStage::PreAttention );

    for ( size_t i = 0; i < batch_size_; ++i ) {
      auto& entry = active_prompts_.emplace_back( std::move( prompt_queue_.front() ) );
      prompt_queue_.pop();
      active_contexts_.push_back( make_shared<typename Model::ContextType>( model_.settings() ) );
      state_.set_prompt( i, entry.id(), ContextID {}, entry.prompt().at( 0 ), 0, temp_, entry.prompt().count(), 0, 0, 0 );
    }
  }

  void run()
  {
    constexpr auto status_interval = chrono::seconds( 5 );
    auto last_status_print = chrono::steady_clock::now();

    while ( true ) {
      /* print the status every `status_interval` seconds */
      if ( const auto now = chrono::steady_clock::now(); now - last_status_print > status_interval ) {
        last_status_print = now;
        LOG( INFO ) << "queued: " << prompt_queue_.size() << " active: " << active_prompts_.size()
                    << " finished: " << finished_prompts_.size();
      }

      for ( size_t layer = 0; layer < Model::ConfigType::n_layers; layer++ ) {
        model_.forward_pre_attention( state_ );
        model_.forward_attention( state_, active_contexts_ );
        model_.forward_post_attention( state_ );

        if ( state_.next_stage() == InferenceStage::Classification ) {
          model_.forward_classify( state_ );
        }
      }

      bool any_active = false;
      for ( size_t i = 0; i < batch_size_; i++ ) {
        if ( not state_.active( i ) ) {
          continue;
        }

        any_active = true;

        if ( state_.token_pos( i ) < state_.prompt_length( i ) ) {
          state_.set_token( i, active_prompts_[i].prompt().at( state_.token_pos( i ) ) );
        } else {
          active_prompts_[i].completion().append( state_.token( i ) );
        }

        if ( state_.finished( i ) ) {
          completions_file_ << active_prompts_[i].to_json() << endl;
          finished_prompts_.push_back( std::move( active_prompts_[i] ) );

          // do we have a prompt in the queue to replace this one?
          if ( !prompt_queue_.empty() ) {
            active_prompts_[i] = std::move( prompt_queue_.front() );
            prompt_queue_.pop();

            auto& entry = active_prompts_[i];
            state_.set_prompt(
              i, entry.id(), state_.context_id( i ), entry.prompt().at( 0 ), 0, temp_, entry.prompt().count(), 0, 0, 0 );
          } else {
            state_.discard( i );
          }
        }
      }

      if ( not any_active ) {
        // all prompts are finished
        break;
      }
    }
  }
};

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0
       << " <model_dir> <model_name> (paged|static)  <tokenizer_path> <batch_size> <temperature> <in:prompts.jsonl> "
          "<out:completions.jsonl>"
       << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 9 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir_path { argv[1] };
    const string model_name { argv[2] };
    const string context_name { argv[3] };
    const filesystem::path tokenizer_path { argv[4] };
    const size_t batch_size = atoi( argv[5] );
    const float temp = atof( argv[6] );
    const filesystem::path prompts_path { argv[7] };
    const filesystem::path completions_path { argv[8] };

#define CREATE_AND_RUN( MODEL_NAME, CONTEXT_NAME, CLASS_NAME )                                                         \
  if ( model_name == MODEL_NAME and context_name == CONTEXT_NAME ) {                                                   \
    using ModelType = llama2::_ORTHRUS_ARCH_NS_::CLASS_NAME<_ORTHRUS_DTYPE_>;                                      \
    BatchInference<ModelType> inference(                                                                               \
      model_dir_path, tokenizer_path, prompts_path, completions_path, batch_size, temp );                              \
    inference.run();                                                                                                   \
  }

    // XXX(): ugly af
    // clang-format off
    CREATE_AND_RUN( "llama2-7b-chat", "static", Llama2_7B_Chat_Static )
    else CREATE_AND_RUN( "llama2-13b-chat", "static", Llama2_13B_Chat_Static )
    else CREATE_AND_RUN( "llama2-70b-chat", "static", Llama2_70B_Chat_Static )
    else CREATE_AND_RUN( "stories-110m", "static", Stories_110M_Static )
    else CREATE_AND_RUN( "llama2-7b-chat", "paged", Llama2_7B_Chat_Paged )
    else CREATE_AND_RUN( "llama2-13b-chat", "paged", Llama2_13B_Chat_Paged )
    else CREATE_AND_RUN( "llama2-70b-chat", "paged", Llama2_70B_Chat_Paged )
    else CREATE_AND_RUN( "stories-110m", "paged", Stories_110M_Paged )
    else LOG( FATAL ) << "Unknown model name: " << model_name << ", or context name: " << context_name;
    // clang-format on

    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
