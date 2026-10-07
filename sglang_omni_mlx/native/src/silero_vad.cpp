// SPDX-License-Identifier: Apache-2.0
#include "silero_vad.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "swift_port.h"

namespace silero_vad {

namespace mx = mlx::core;

namespace {

using qwen3_asr::kSampleRate;
// Speech decisions are made on blocks of eight 512-sample chunks, 256 ms each.
constexpr int kChunkSampleCount = 512;
constexpr int kChunksPerBlock = 8;
// Evaluated every 16 chunks, as the Swift port does, so the graph stays small.
constexpr size_t kChunksPerEvaluation = 16;

} // namespace

SileroVad::SileroVad(const std::filesystem::path &model_directory) {
  std::ifstream config_stream(model_directory / "config.json");
  if (!config_stream) {
    throw std::runtime_error("cannot read config.json in " +
                             model_directory.string());
  } else {
  }
  const nlohmann::json branch =
      nlohmann::json::parse(config_stream).at("branch_16k");
  filter_hop_length_ = branch.at("hop_length").get<int>();
  reflect_padding_ = branch.at("pad").get<int>();
  frequency_cutoff_ = branch.at("cutoff").get<int>();
  context_sample_count_ = branch.at("context_size").get<int>();
  chunk_sample_count_ = branch.at("chunk_size").get<int>();
  const std::string branch_prefix = "vad_16k.";
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() != ".safetensors") {
      continue;
    } else {
    }
    auto [loaded, metadata] = mx::load_safetensors(entry.path().string());
    for (auto &[name, array] : loaded) {
      if (name.rfind(branch_prefix, 0) == 0) {
        weights_.insert_or_assign(name.substr(branch_prefix.size()), array);
      } else {
      }
    }
  }
  if (weights_.empty()) {
    throw std::runtime_error("Silero VAD checkpoint has no weights");
  }
  std::vector<mx::array> parameters;
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &SileroVad::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("Silero VAD checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

mx::array SileroVad::Conv1d(const mx::array &x, const std::string &prefix,
                            int stride, int padding) const {
  return mx::add(mx::conv1d(x, Weight(prefix + ".weight"), stride, padding),
                 Weight(prefix + ".bias"));
}

mx::array SileroVad::ChunkProbability(const mx::array &window,
                                      std::optional<mx::array> &state) const {
  // Reflect padding on the right: samples n - 2 down to n - padding - 1.
  const int sample_count = window.shape(-1);
  std::vector<int32_t> reflected_indices(reflect_padding_);
  for (int i = 0; i < reflect_padding_; ++i)
    reflected_indices[i] = sample_count - 2 - i;
  mx::array x = mx::concatenate(
      {window, mx::take(window,
                        mx::array(reflected_indices.data(), {reflect_padding_},
                                  mx::int32),
                        -1)},
      -1);
  // The STFT as a convolution: real and imaginary halves to magnitudes.
  x = mx::conv1d(mx::expand_dims(x, -1), Weight("stft_conv.weight"),
                 filter_hop_length_, 0);
  const mx::array real =
      mx::slice(x, {0, 0, 0}, {x.shape(0), x.shape(1), frequency_cutoff_});
  const mx::array imaginary =
      mx::slice(x, {0, 0, frequency_cutoff_},
                {x.shape(0), x.shape(1), 2 * frequency_cutoff_});
  x = mx::sqrt(
      mx::add(mx::multiply(real, real), mx::multiply(imaginary, imaginary)));
  x = swift_port::Relu(Conv1d(x, "conv1", 1, 1));
  x = swift_port::Relu(Conv1d(x, "conv2", 2, 1));
  x = swift_port::Relu(Conv1d(x, "conv3", 2, 1));
  x = swift_port::Relu(Conv1d(x, "conv4", 1, 1));

  // One LSTM layer over the frames, from the previous window's state.
  const mx::array projected =
      mx::addmm(Weight("lstm.bias"), x, mx::transpose(Weight("lstm.Wx")));
  std::optional<mx::array> hidden;
  std::optional<mx::array> cell;
  if (state.has_value()) {
    hidden = mx::take(*state, 0, 0);
    cell = mx::take(*state, 1, 0);
  } else {
  }
  std::vector<mx::array> hidden_states;
  std::vector<mx::array> cell_states;
  const int frame_count = projected.shape(1);
  const int gate_width = projected.shape(2);
  for (int frame = 0; frame < frame_count; ++frame) {
    mx::array gates =
        mx::reshape(mx::slice(projected, {0, frame, 0},
                              {projected.shape(0), frame + 1, gate_width}),
                    {projected.shape(0), gate_width});
    if (hidden.has_value()) {
      gates = mx::addmm(gates, *hidden, mx::transpose(Weight("lstm.Wh")));
    } else {
    }
    const std::vector<mx::array> pieces = mx::split(gates, 4, -1);
    const mx::array input_gate = mx::sigmoid(pieces[0]);
    const mx::array forget_gate = mx::sigmoid(pieces[1]);
    const mx::array candidate = mx::tanh(pieces[2]);
    const mx::array output_gate = mx::sigmoid(pieces[3]);
    if (cell.has_value()) {
      cell = mx::add(mx::multiply(forget_gate, *cell),
                     mx::multiply(input_gate, candidate));
    } else {
      cell = mx::multiply(input_gate, candidate);
    }
    hidden = mx::multiply(output_gate, mx::tanh(*cell));
    hidden_states.push_back(*hidden);
    cell_states.push_back(*cell);
  }
  const mx::array hidden_sequence = mx::stack(hidden_states, -2);
  state = mx::stack({hidden_states.back(), cell_states.back()}, 0);
  const mx::array frame_probabilities = mx::sigmoid(
      Conv1d(swift_port::Relu(hidden_sequence), "final_conv", 1, 0));
  return mx::mean(mx::squeeze(frame_probabilities, -1), 1, true);
}

std::vector<float>
SileroVad::SpeechProbabilities(const std::vector<float> &samples) const {
  if (samples.empty()) {
    return {};
  } else {
  }
  // Silence pads the last chunk; context_sample_count zeros precede the first.
  const size_t padding =
      (chunk_sample_count_ - samples.size() % chunk_sample_count_) %
      chunk_sample_count_;
  std::vector<float> padded(context_sample_count_, 0.0f);
  padded.insert(padded.end(), samples.begin(), samples.end());
  padded.resize(padded.size() + padding, 0.0f);
  const mx::array audio(padded.data(), {1, static_cast<int>(padded.size())},
                        mx::float32);
  std::vector<mx::array> probabilities;
  std::optional<mx::array> state;
  for (size_t position = context_sample_count_; position < padded.size();
       position += chunk_sample_count_) {
    const mx::array window = mx::slice(
        audio, {0, static_cast<int>(position) - context_sample_count_},
        {1, static_cast<int>(position) + chunk_sample_count_});
    probabilities.push_back(ChunkProbability(window, state));
    if (probabilities.size() % kChunksPerEvaluation == 0) {
      mx::async_eval({probabilities.back(), *state});
    } else {
    }
  }
  const mx::array all_probabilities =
      mx::reshape(mx::concatenate(probabilities, 1), {-1});
  mx::eval(all_probabilities);
  const float *values = all_probabilities.data<float>();
  return std::vector<float>(values, values + all_probabilities.size());
}

std::vector<std::pair<size_t, size_t>>
SegmentSpeech(const SileroVad &vad, const std::vector<float> &samples,
              const SpeechSegmentConfig &config) {
  const std::vector<float> chunk_probabilities =
      vad.SpeechProbabilities(samples);
  const int block_samples = kChunkSampleCount * kChunksPerBlock;
  const float block_seconds =
      static_cast<float>(block_samples) / static_cast<float>(kSampleRate);
  const int block_count =
      static_cast<int>(chunk_probabilities.size() / kChunksPerBlock);
  // A block is speech unless every one of its chunks is silent.
  std::vector<float> block_probabilities(block_count);
  for (int block = 0; block < block_count; ++block) {
    float silence = 1.0f;
    for (int chunk = 0; chunk < kChunksPerBlock; ++chunk) {
      silence *= 1.0f - chunk_probabilities[block * kChunksPerBlock + chunk];
    }
    block_probabilities[block] = 1.0f - silence;
  }
  const auto blocks_of = [&](int milliseconds) {
    return static_cast<float>(milliseconds) / 1000.0f / block_seconds;
  };
  const int speech_pad_blocks =
      std::max(0, static_cast<int>(blocks_of(config.speech_pad_milliseconds)));
  // Minimum durations round up: 500 ms takes two blocks, not one.
  const int min_speech_blocks = std::max(
      1,
      static_cast<int>(std::ceil(blocks_of(config.min_speech_milliseconds))));
  const int min_silence_blocks = std::max(
      1,
      static_cast<int>(std::ceil(blocks_of(config.min_silence_milliseconds))));

  std::vector<std::pair<size_t, size_t>> runs;
  int segment_start = 0;
  // The open run's last speech block; -1 when no run is open.
  int last_speech = -1;
  const auto close_run = [&]() {
    const int segment_end =
        std::min(last_speech + 1 + speech_pad_blocks, block_count);
    const size_t start = static_cast<size_t>(segment_start) * block_samples;
    const size_t end = std::min(
        static_cast<size_t>(segment_end) * block_samples, samples.size());
    if (segment_end - segment_start >= min_speech_blocks && start < end) {
      runs.emplace_back(start, end);
    } else {
    }
    last_speech = -1;
  };
  for (int block = 0; block < block_count; ++block) {
    if (block_probabilities[block] >= config.threshold) {
      if (last_speech < 0) {
        segment_start = std::max(0, block - speech_pad_blocks);
      } else {
      }
      last_speech = block;
    } else if (last_speech >= 0 && block - last_speech >= min_silence_blocks) {
      close_run();
    } else {
    }
  }
  if (last_speech >= 0) {
    close_run();
  } else {
  }

  // Runs closer than the merge gap join while they fit one chunk; longer
  // runs split at the chunk length.
  const size_t max_chunk_samples = static_cast<size_t>(
      std::max(1, static_cast<int>(config.max_chunk_seconds *
                                   static_cast<float>(kSampleRate))));
  const long max_gap_samples = static_cast<long>(
      config.merge_gap_seconds * static_cast<float>(kSampleRate));
  std::vector<std::pair<size_t, size_t>> chunks;
  for (const auto &[start, end] : runs) {
    if (!chunks.empty() &&
        static_cast<long>(start) - static_cast<long>(chunks.back().second) <=
            max_gap_samples &&
        end - chunks.back().first <= max_chunk_samples) {
      chunks.back().second = end;
    } else {
      for (size_t cut = start; cut < end;
           cut = std::min(cut + max_chunk_samples, end)) {
        chunks.emplace_back(cut, std::min(cut + max_chunk_samples, end));
      }
    }
  }
  return chunks;
}

} // namespace silero_vad
