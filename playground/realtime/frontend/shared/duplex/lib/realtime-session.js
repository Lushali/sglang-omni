/** Native SGLang-Omni session; media capture and buffered playback stay in the UI. */
import { AudioPlayer } from './audio-player.js';
import { arrayBufferToBase64 } from './duplex-utils.js';

export function pcm16(samples) {
    const bytes = new Uint8Array(samples.length * 2);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < samples.length; i++) {
        const value = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
    }
    return bytes;
}

export function referenceWav(encoded) {
    const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    const samples = new Float32Array(bytes.buffer);
    const pcm = pcm16(samples);
    if (pcm.length + 44 > 1024 * 1024) throw new Error('Reference WAV exceeds the 1 MiB server limit.');
    const wav = new Uint8Array(44 + pcm.length);
    const view = new DataView(wav.buffer);
    const text = (offset, value) => [...value].forEach((character, i) => view.setUint8(offset + i, character.charCodeAt(0)));
    text(0, 'RIFF'); view.setUint32(4, 36 + pcm.length, true); text(8, 'WAVE');
    text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, 'data'); view.setUint32(40, pcm.length, true); wav.set(pcm, 44);
    return {media_type: 'audio/wav', data: arrayBufferToBase64(wav.buffer)};
}

export function flushCapture(node) {
    if (!node) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => { node.port.removeEventListener('message', receive); reject(new Error('Microphone flush timed out')); }, 2000);
        function receive(event) {
            if (event.data.type !== 'stopped') return;
            clearTimeout(timeout);
            node.port.removeEventListener('message', receive);
            resolve();
        }
        node.port.addEventListener('message', receive);
        node.port.postMessage({command: 'stop'});
    });
}

export class RealtimeSession {
    constructor(prefix, config = {}) {
        this.prefix = prefix;
        this.config = config;
        this.audioPlayer = new AudioPlayer({outputSampleRate: 24000, getPlaybackDelayMs: config.getPlaybackDelayMs});
        this.audioPlayer.onMetrics = metrics => this.onMetrics({type: 'audio', ...metrics});
        this.ws = null;
        this.sessionId = '';
        this.backendInfo = null;
        this.chunksSent = 0;
        this.samplesSent = 0;
        this.sequence = 0;
        this.eventSequence = 0;
        this._started = false;
        this.stopping = false;
        this.closed = false;
        this.cleaned = false;
        this.paused = false;
        this._eventLog = [];
        this._sessionStartTime = 0;
        this._resultCount = 0;
        this._lastDriftMs = null;
        this.currentSpeakText = '';
        this.speakHandle = null;
        this.responseId = null;
        this.unitsWithOutput = new Set();
    }
    get running() { return this._started; }
    get eventLog() { return this._eventLog; }
    onSystemLog(text) {}
    onSpeakStart(text) { return null; }
    onSpeakUpdate(handle, text) {}
    onSpeakEnd() {}
    onListenResult(result) {}
    onExtraResult(result, time) {}
    async onPrepared() {}
    async onBeforeStop() {}
    onCleanup() {}
    onMetrics(metrics) {}
    onRunningChange(running) {}
    onProtocolEvent(event) {}

    log(direction, event) {
        const safe = {...event};
        if (safe.audio) safe.audio = '<PCM16 audio>';
        if (safe.image) safe.image = '<image>';
        if (safe.type === 'response.output_audio.delta') safe.delta = '<PCM16 audio>';
        if (safe.session?.sglang) {
            safe.session = {...safe.session, sglang: {...safe.session.sglang}};
            for (const field of ['reference_audio', 'tts_reference_audio']) {
                if (safe.session.sglang[field]) safe.session.sglang[field] = {media_type: 'audio/wav', data: '<WAV audio>'};
            }
        }
        const entry = {ts: Date.now(), dir: direction, type: event.type, summary: event.type, full: safe};
        this._eventLog.push(entry);
        if (this._eventLog.length > 200) this._eventLog.shift();
        this.onProtocolEvent(entry);
    }
    send(type, fields = {}) {
        if (this.ws?.readyState !== WebSocket.OPEN) throw new Error('Realtime connection is closed');
        const event = {type, event_id: 'client-' + this.eventSequence++, ...fields};
        this.ws.send(JSON.stringify(event));
        this.log('client', event);
    }

    async start(instructions, options, startMedia) {
        this._sessionStartTime = performance.now();
        this.audioPlayer.init();
        try {
            const response = await fetch('/v1/realtime/capabilities');
            if (!response.ok) throw new Error('Model service unavailable: ' + response.status);
            const capabilities = await response.json();
            if (!capabilities.native_full_duplex || capabilities.input_audio_format.rate !== 16000 || capabilities.output_audio_format.rate !== 24000) {
                throw new Error('This MiniCPM page requires native duplex with 16 kHz input and 24 kHz output.');
            }
            const extension = {sampling: options.sampling || {}};
            for (const key of Object.keys(extension.sampling)) {
                if (!capabilities.sampling_parameters.includes(key)) throw new Error('Unsupported sampling option: ' + key);
            }
            if (options.max_slice_nums) extension.max_slice_nums = options.max_slice_nums;
            for (const field of ['reference_audio', 'tts_reference_audio']) {
                if (!options[field]) continue;
                if (!capabilities.supports_reference_audio) throw new Error('Reference audio is not supported by this deployment');
                extension[field] = referenceWav(options[field]);
            }
            const session = {instructions, output_modalities: options.use_tts === false ? ['text'] : ['audio'], audio: {input: {format: capabilities.input_audio_format, turn_detection: null}}, sglang: extension};
            if (this.cleaned) throw new Error('Session cancelled');
            await new Promise((resolve, reject) => {
                this.rejectStart = reject;
                this.timeout = setTimeout(() => reject(new Error('Session initialization timed out')), 60000);
                const address = new URL('/v1/realtime', location.href);
                address.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                this.ws = new WebSocket(address);
                this.ws.onmessage = message => {
                    try {
                        const event = JSON.parse(message.data);
                        this.log('server', event);
                        if (event.type === 'session.created') {
                            this.sessionId = event.session.id;
                            this.send('session.update', {session});
                        } else if (event.type === 'session.updated') {
                            clearTimeout(this.timeout);
                            this.granted = event.session.sglang.granted;
                            if (this.granted.rejections.length) throw new Error(this.granted.rejections.map(entry => entry.field + ': ' + entry.reason).join('; '));
                            this.backendInfo = {fixed_settings: ['max_slice_nums'], max_slice_nums: extension.max_slice_nums || 1, sampling: extension.sampling};
                            this.rejectStart = null;
                            this.onMetrics({type: 'state', sessionId: this.sessionId});
                            resolve();
                        } else if (event.type === 'error') {
                            throw new Error(event.error?.message || JSON.stringify(event));
                        } else {
                            this.handle(event);
                        }
                    } catch (error) { this.fail(error); }
                };
                this.ws.onerror = () => this.fail(new Error('Realtime WebSocket connection failed'));
                this.ws.onclose = () => {
                    if (!this.closed && !this.cleaned) this.fail(new Error('Server disconnected before session.closed'));
                };
            });
            await this.onPrepared();
            if (this.cleaned) throw new Error('Session cancelled');
            this._started = true;
            if (startMedia) await startMedia();
            if (this.cleaned) { this.onCleanup(); throw new Error('Session cancelled'); }
            this.onRunningChange(true);
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    sendChunk({audio, frames = []}) {
        if (!this._started || this.inputEnded) return;
        try {
            if (this.ws.bufferedAmount > 1024 * 1024) throw new Error('Network cannot keep up with captured audio');
            const timeMs = this.samplesSent / 16;
            if (frames.length) {
                const format = this.granted.input_image_format;
                if (!format || frames.length > format.max_per_unit) throw new Error('Image count exceeds deployment capability');
                for (const image of frames) {
                    if (image.length * 3 / 4 > format.max_bytes) throw new Error('Image exceeds deployment byte limit');
                    this.send('sglang.input_image.append', {image, sglang: {t_ms: timeMs}});
                }
            }
            const bytes = pcm16(audio);
            for (let offset = 0; offset < bytes.length; offset += 2560) {
                const packet = bytes.slice(offset, offset + 2560);
                this.send('input_audio_buffer.append', {audio: arrayBufferToBase64(packet.buffer), sglang: {seq: this.sequence++, t_start_ms: this.samplesSent / 16}});
                this.samplesSent += packet.length / 2;
            }
            this.chunksSent++;
            this.onMetrics({type: 'result', chunksSent: this.chunksSent});
        } catch (error) { this.fail(error); }
    }

    handle(event) {
        if (event.type === 'response.created') {
            this.responseId = event.response.id;
            this.currentSpeakText = '';
            this.speakHandle = null;
        } else if (event.type === 'response.output_audio.delta') {
            const bytes = Uint8Array.from(atob(event.delta), character => character.charCodeAt(0));
            if (bytes.length % 2) throw new Error('Incomplete PCM16 output');
            const view = new DataView(bytes.buffer);
            const audio = new Float32Array(bytes.length / 2);
            for (let i = 0; i < audio.length; i++) audio[i] = view.getInt16(i * 2, true) / 32768;
            if (!this.audioPlayer.turnActive) this.audioPlayer.beginTurn();
            this.audioPlayer.playChunk(arrayBufferToBase64(audio.buffer), performance.now());
            this.unitsWithOutput.add(event.sglang?.unit_id);
            this._resultCount++;
            this.onExtraResult({is_listen: false, text: '', end_of_turn: false}, performance.now());
            this.onMetrics({type: 'result', modelState: 'speaking', chunksSent: this.chunksSent});
        } else if (event.type === 'response.output_audio_transcript.delta' || event.type === 'response.output_text.delta') {
            this.unitsWithOutput.add(event.sglang?.unit_id);
            this.currentSpeakText += event.delta;
            if (!this.speakHandle) this.speakHandle = this.onSpeakStart(this.currentSpeakText);
            else this.onSpeakUpdate(this.speakHandle, this.currentSpeakText);
        } else if (event.type === 'response.done') {
            this.audioPlayer.endTurn();
            this.onSpeakEnd();
            this.speakHandle = null;
            this.onMetrics({type: 'result', modelState: 'listening'});
        } else if (event.type === 'sglang.unit.done') {
            if (!this.unitsWithOutput.delete(event.unit_id)) this.onListenResult({is_listen: true});
        } else if (event.type === 'sglang.input_audio.drained') {
            this.send('session.close');
        } else if (event.type === 'session.closed') {
            if (this.rejectStart) throw new Error(event.diagnostic?.message || 'Session closed during initialization');
            this.closed = true;
            clearTimeout(this.timeout);
            this.audioPlayer.endTurn();
            this.onSystemLog('Session closed: ' + this.sessionId);
            this.timeout = setTimeout(() => this.cleanup(), Math.max(0, this.audioPlayer.lastAheadMs) + 100);
        }
    }

    async stop() {
        if (this.stopping || this.cleaned) return;
        this.stopping = true;
        if (!this._started) { this.cancelStart(); return; }
        this.onMetrics({type: 'state', sessionState: 'Finishing...'});
        try {
            await this.onBeforeStop();
            if (this.cleaned) return;
            this.inputEnded = true;
            this.send('sglang.input_audio.end');
            this.timeout = setTimeout(() => this.fail(new Error('Session drain timed out')), 30000);
        } catch (error) { this.fail(error); }
    }
    cancelStart() { this.fail(new Error('Session cancelled')); }
    fail(error) {
        if (this.cleaned) return;
        this.onSystemLog('Error: ' + error.message);
        this.rejectStart?.(error);
        this.rejectStart = null;
        this.cleanup();
    }
    cleanup() {
        if (this.cleaned) return;
        this.cleaned = true;
        clearTimeout(this.timeout);
        this._started = false;
        this.onCleanup();
        this.audioPlayer.stop();
        this.ws?.close();
        this.onRunningChange(false);
        this.onMetrics({type: 'state', sessionState: 'Stopped'});
    }
}
