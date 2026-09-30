## main-a
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.27 | 0.27 |
  | preprocessing | stage_complete | 4.62 | 4.32 |
  | tts_engine | scheduler_request_build_start | 4.9 | 0.31 |
  | tts_engine | scheduler_request_build_end | 5.21 | 0.31 |
  | tts_engine | scheduler_queue_enter | 5.33 | 0.12 |
  | tts_engine | scheduler_prefill_start | 6.38 | 1.04 |
  | tts_engine | scheduler_prefill_end | 12.44 | 5.9 |
  | tts_engine | scheduler_first_emit | 12.56 | 0.12 |
  | vocoder | stage_stream_chunk_received | 13.35 | 0.85 |
  | vocoder | stage_first_stream_chunk_sent | 44.5 | 30.12 |
  | coordinator | coordinator_stream_received | 44.78 | 0.22 |

## main-b
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.26 | 0.26 |
  | preprocessing | stage_complete | 4.52 | 4.23 |
  | tts_engine | scheduler_request_build_start | 4.81 | 0.29 |
  | tts_engine | scheduler_request_build_end | 5.15 | 0.31 |
  | tts_engine | scheduler_queue_enter | 5.28 | 0.12 |
  | tts_engine | scheduler_prefill_start | 6.28 | 1.02 |
  | tts_engine | scheduler_prefill_end | 12.21 | 5.73 |
  | tts_engine | scheduler_first_emit | 12.33 | 0.12 |
  | vocoder | stage_stream_chunk_received | 13.19 | 0.83 |
  | vocoder | stage_first_stream_chunk_sent | 43.45 | 30.16 |
  | coordinator | coordinator_stream_received | 43.65 | 0.21 |

## prime-a
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.25 | 0.25 |
  | preprocessing | stage_complete | 4.53 | 4.27 |
  | tts_engine | scheduler_request_build_start | 4.88 | 0.29 |
  | tts_engine | scheduler_request_build_end | 5.19 | 0.3 |
  | tts_engine | scheduler_queue_enter | 5.3 | 0.11 |
  | tts_engine | scheduler_prefill_start | 6.33 | 0.98 |
  | tts_engine | scheduler_prefill_end | 12.2 | 5.71 |
  | tts_engine | scheduler_first_emit | 12.31 | 0.12 |
  | vocoder | stage_stream_chunk_received | 13.23 | 0.88 |
  | vocoder | stage_first_stream_chunk_sent | 40.37 | 27.03 |
  | coordinator | coordinator_stream_received | 40.6 | 0.21 |

## prime-b
  ### events-hot (60 timed requests)
  | stage | event | offset p50 ms | gap p50 ms |
  | coordinator | request_admission | 0.0 | None |
  | preprocessing | stage_input_received | 0.3 | 0.3 |
  | preprocessing | stage_complete | 4.62 | 4.29 |
  | tts_engine | scheduler_request_build_start | 4.96 | 0.31 |
  | tts_engine | scheduler_request_build_end | 5.26 | 0.31 |
  | tts_engine | scheduler_queue_enter | 5.37 | 0.12 |
  | tts_engine | scheduler_prefill_start | 6.4 | 1.03 |
  | tts_engine | scheduler_prefill_end | 12.19 | 5.73 |
  | tts_engine | scheduler_first_emit | 12.31 | 0.12 |
  | vocoder | stage_stream_chunk_received | 13.19 | 0.83 |
  | vocoder | stage_first_stream_chunk_sent | 40.63 | 27.11 |
  | coordinator | coordinator_stream_received | 40.89 | 0.24 |

