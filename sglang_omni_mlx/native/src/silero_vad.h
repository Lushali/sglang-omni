// SPDX-License-Identifier: Apache-2.0
// Silero VAD v6 on MLX, and the speech segmentation Voxt runs before Cohere
// Transcribe on long audio, op for op as Voxt's Swift port.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/mlx.h"

namespace silero_vad {

// How speech runs become chunks: Voxt's long-form settings.
struct SpeechSegmentConfig {
  float threshold = 0.0f;
  int min_speech_milliseconds = 0;
  int min_silence_milliseconds = 0;
  int speech_pad_milliseconds = 0;
  float merge_gap_seconds = 0.0f;
  float max_chunk_seconds = 0.0f;
};

class SileroVad {
public:
  // Reads config.json and the 16 kHz branch of model.safetensors.
  explicit SileroVad(const std::filesystem::path &model_directory);

  // The speech probability of every 512-sample chunk of 16 kHz samples, the
  // last chunk padded with silence.
  std::vector<float>
  SpeechProbabilities(const std::vector<float> &samples) const;

private:
  const mlx::core::array &Weight(const std::string &name) const;
  mlx::core::array Conv1d(const mlx::core::array &x, const std::string &prefix,
                          int stride, int padding) const;
  // One window of context and chunk samples [1, samples] to its speech
  // probability [1, 1]; state carries the LSTM's hidden and cell [2, 1, 128].
  mlx::core::array
  ChunkProbability(const mlx::core::array &window,
                   std::optional<mlx::core::array> &state) const;

  int filter_hop_length_ = 0;
  int reflect_padding_ = 0;
  int frequency_cutoff_ = 0;
  int context_sample_count_ = 0;
  int chunk_sample_count_ = 0;
  std::unordered_map<std::string, mlx::core::array> weights_;
};

// [start, end) sample ranges of the speech in 16 kHz samples: runs of
// speech blocks, merged across short gaps and split at the longest chunk.
// Empty when there is no speech.
std::vector<std::pair<size_t, size_t>>
SegmentSpeech(const SileroVad &vad, const std::vector<float> &samples,
              const SpeechSegmentConfig &config);

} // namespace silero_vad
