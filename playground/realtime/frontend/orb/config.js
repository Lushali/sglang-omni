// The playground server proxies /v1/realtime on the page's own origin.
window.DEMO_WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/v1/realtime`;
window.DEMO_STATUS_URL = "";
window.DEMO_MODEL_NAME = "MiniCPM-o 4.5";
// Optional playback policy, e.g. { maxLagMs: 1500, speedUp: true }: audio further behind
// the model than maxLagMs is dropped. Users can override both under Settings > Advanced.
// window.DEMO_PLAYBACK = {};
// Persona sent as the session's system prompt. Without one the thinker (a Qwen3
// derivative) introduces itself as Qwen.
window.DEMO_INSTRUCTIONS = "You are MiniCPM-o 4.5, a friendly voice assistant built by OpenBMB and served by sglang-omni. Answer briefly in the language the user speaks. If asked to count or list many items, give only the first few.";
