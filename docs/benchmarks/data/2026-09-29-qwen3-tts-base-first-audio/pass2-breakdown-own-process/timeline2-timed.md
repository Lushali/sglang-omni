## base-icl
  r1-plain: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0549, 'audio_ttfp_from_arrival_p95_s': 0.0707, 'client_slot_waits': 0}
  r1-events-cold: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0538, 'audio_ttfp_from_arrival_p95_s': 0.0749, 'client_slot_waits': 0}
  r1-events-hot: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0488, 'audio_ttfp_from_arrival_p95_s': 0.0696, 'client_slot_waits': 0}
  r20-plain: {'completed_requests': 1088, 'audio_ttfp_from_arrival_median_s': 0.1035, 'audio_ttfp_from_arrival_p95_s': 0.3815, 'client_slot_waits': 0}
  ### events-cold (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.3 | 0.3 |
  | preprocessing | stage_complete | 10.63 | 10.36 |
  | tts_engine | scheduler_request_build_start | 10.98 | 0.29 |
  | tts_engine | scheduler_request_build_end | 11.28 | 0.31 |
  | tts_engine | scheduler_queue_enter | 11.4 | 0.12 |
  | tts_engine | scheduler_prefill_start | 12.45 | 1.07 |
  | tts_engine | scheduler_prefill_end | 19.26 | 6.68 |
  | tts_engine | scheduler_first_emit | 19.38 | 0.12 |
  | vocoder | stage_stream_chunk_received | 20.38 | 0.85 |
  | vocoder | stage_first_stream_chunk_sent | 50.63 | 30.4 |
  | coordinator | coordinator_stream_received | 50.83 | 0.22 |
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.27 | 0.27 |
  | preprocessing | stage_complete | 4.79 | 4.48 |
  | tts_engine | scheduler_request_build_start | 5.09 | 0.3 |
  | tts_engine | scheduler_request_build_end | 5.46 | 0.32 |
  | tts_engine | scheduler_queue_enter | 5.57 | 0.12 |
  | tts_engine | scheduler_prefill_start | 6.67 | 1.06 |
  | tts_engine | scheduler_prefill_end | 12.59 | 5.81 |
  | tts_engine | scheduler_first_emit | 12.72 | 0.12 |
  | vocoder | stage_stream_chunk_received | 13.53 | 0.84 |
  | vocoder | stage_first_stream_chunk_sent | 46.1 | 30.32 |
  | coordinator | coordinator_stream_received | 46.32 | 0.21 |

## base-xvec
  r1-plain: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0327, 'audio_ttfp_from_arrival_p95_s': 0.0417, 'client_slot_waits': 0}
  r1-events-cold: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0304, 'audio_ttfp_from_arrival_p95_s': 0.0395, 'client_slot_waits': 0}
  r1-events-hot: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0246, 'audio_ttfp_from_arrival_p95_s': 0.0303, 'client_slot_waits': 0}
  ### events-cold (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.27 | 0.27 |
  | preprocessing | stage_complete | 8.1 | 7.87 |
  | tts_engine | scheduler_request_build_start | 8.47 | 0.3 |
  | tts_engine | scheduler_request_build_end | 8.81 | 0.32 |
  | tts_engine | scheduler_queue_enter | 8.93 | 0.12 |
  | tts_engine | scheduler_prefill_start | 9.96 | 1.02 |
  | tts_engine | scheduler_prefill_end | 16.76 | 6.55 |
  | tts_engine | scheduler_first_emit | 16.84 | 0.07 |
  | vocoder | stage_stream_chunk_received | 17.58 | 0.85 |
  | vocoder | stage_first_stream_chunk_sent | 26.9 | 9.22 |
  | coordinator | coordinator_stream_received | 27.13 | 0.24 |
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.27 | 0.27 |
  | preprocessing | stage_complete | 2.8 | 2.53 |
  | tts_engine | scheduler_request_build_start | 3.1 | 0.29 |
  | tts_engine | scheduler_request_build_end | 3.41 | 0.31 |
  | tts_engine | scheduler_queue_enter | 3.52 | 0.11 |
  | tts_engine | scheduler_prefill_start | 4.49 | 0.98 |
  | tts_engine | scheduler_prefill_end | 10.39 | 5.75 |
  | tts_engine | scheduler_first_emit | 10.46 | 0.07 |
  | vocoder | stage_stream_chunk_received | 11.41 | 0.89 |
  | vocoder | stage_first_stream_chunk_sent | 20.66 | 9.2 |
  | coordinator | coordinator_stream_received | 20.92 | 0.22 |

## cv
  r1-plain: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.025, 'audio_ttfp_from_arrival_p95_s': 0.0313, 'client_slot_waits': 0}
  r1-events-cold: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0269, 'audio_ttfp_from_arrival_p95_s': 0.329, 'client_slot_waits': 0}
  r1-events-hot: {'completed_requests': 60, 'audio_ttfp_from_arrival_median_s': 0.0239, 'audio_ttfp_from_arrival_p95_s': 0.0307, 'client_slot_waits': 0}
  r20-plain: {'completed_requests': 1088, 'audio_ttfp_from_arrival_median_s': 0.0383, 'audio_ttfp_from_arrival_p95_s': 0.362, 'client_slot_waits': 0}
  ### events-cold (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.26 | 0.26 |
  | preprocessing | stage_complete | 2.75 | 2.46 |
  | tts_engine | scheduler_request_build_start | 3.07 | 0.31 |
  | tts_engine | scheduler_request_build_end | 3.41 | 0.32 |
  | tts_engine | scheduler_queue_enter | 3.53 | 0.12 |
  | tts_engine | scheduler_prefill_start | 4.64 | 1.07 |
  | tts_engine | scheduler_prefill_end | 11.59 | 6.46 |
  | tts_engine | scheduler_first_emit | 11.66 | 0.07 |
  | vocoder | stage_stream_chunk_received | 13.25 | 1.13 |
  | vocoder | stage_first_stream_chunk_sent | 24.01 | 9.22 |
  | coordinator | coordinator_stream_received | 24.2 | 0.21 |
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.25 | 0.25 |
  | preprocessing | stage_complete | 2.68 | 2.43 |
  | tts_engine | scheduler_request_build_start | 2.98 | 0.28 |
  | tts_engine | scheduler_request_build_end | 3.28 | 0.29 |
  | tts_engine | scheduler_queue_enter | 3.39 | 0.11 |
  | tts_engine | scheduler_prefill_start | 4.41 | 1.01 |
  | tts_engine | scheduler_prefill_end | 10.19 | 5.64 |
  | tts_engine | scheduler_first_emit | 10.27 | 0.07 |
  | vocoder | stage_stream_chunk_received | 11.25 | 0.84 |
  | vocoder | stage_first_stream_chunk_sent | 20.46 | 9.26 |
  | coordinator | coordinator_stream_received | 20.65 | 0.2 |

