import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RealtimeSession, pcm16, referenceWav } from '../frontend/shared/duplex/lib/realtime-session.js';

vi.mock('../frontend/shared/duplex/lib/audio-player.js', () => ({
    AudioPlayer: class { lastAheadMs = 0; init() {} stop() {} endTurn() {} },
}));
let reply;
class Socket {
    static OPEN = 1;
    readyState = 1;
    bufferedAmount = 0;
    sent = [];
    constructor() { queueMicrotask(() => this.receive({type: 'session.created', session: {id: 'native'}})); }
    receive(event) { this.onmessage({data: JSON.stringify(event)}); }
    send(raw) {
        const event = JSON.parse(raw);
        this.sent.push(event);
        if (event.type === 'session.update') queueMicrotask(() => this.receive(reply));
    }
    close() { this.readyState = 3; }
}
beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('WebSocket', Socket);
    vi.stubGlobal('location', {href: 'http://localhost/audio_duplex', protocol: 'http:'});
    vi.stubGlobal('fetch', async () => ({ok: true, json: async () => ({native_full_duplex: true,
        input_audio_format: {rate: 16000}, output_audio_format: {rate: 24000},
        sampling_parameters: ['greedy', 'top_k'], supports_reference_audio: true})}));
    reply = {type: 'session.updated', session: {sglang: {granted: {rejections: [], input_image_format: {max_per_unit: 1, max_bytes: 1000}}}}};
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('native duplex session', () => {
    it('encodes PCM16 and WAV reference audio', () => {
        expect([...pcm16(new Float32Array([-1, 0, 1]))]).toEqual([0,128,0,0,255,127]);
        const ref = referenceWav(btoa(String.fromCharCode(...new Uint8Array(new Float32Array([0,1]).buffer))));
        expect(ref.media_type).toBe('audio/wav');
        expect(atob(ref.data).slice(0,4)).toBe('RIFF');
        expect(atob(ref.data).length).toBe(48);
    });
    it('uses native settings and sends images before sequenced PCM16 packets', async () => {
        const session = new RealtimeSession('test', {});
        const sampling = {greedy: false, top_k: 20};
        await session.start('prompt', {sampling});
        expect(session.ws.sent[0]).toMatchObject({type: 'session.update', session: {instructions: 'prompt', sglang: {sampling}}});
        session.sendChunk({audio: new Float32Array(1600), frames: ['AAAA']});
        expect(session.ws.sent.slice(1).map(e => e.type)).toEqual(['sglang.input_image.append','input_audio_buffer.append','input_audio_buffer.append']);
        expect(session.ws.sent[2].sglang).toEqual({seq: 0, t_start_ms: 0});
        expect(session.ws.sent[3].sglang).toEqual({seq: 1, t_start_ms: 80});
        expect(atob(session.ws.sent[3].audio).length).toBe(640);
        session.cleanup();
    });
    it('flushes capture before EOF and waits for drain and close before cleanup', async () => {
        const session = new RealtimeSession('test', {});
        await session.start('', {sampling: {}});
        session.onBeforeStop = async () => session.sendChunk({audio: new Float32Array(10)});
        const cleanup = vi.fn(); session.onCleanup = cleanup;
        await session.stop();
        expect(session.ws.sent.slice(-2).map(e => e.type)).toEqual(['input_audio_buffer.append','sglang.input_audio.end']);
        expect(cleanup).not.toHaveBeenCalled();
        session.ws.receive({type: 'sglang.input_audio.drained'});
        expect(session.ws.sent.at(-1).type).toBe('session.close');
        session.ws.receive({type: 'session.closed'});
        await vi.advanceTimersByTimeAsync(101);
        expect(cleanup).toHaveBeenCalledTimes(1);
        expect(session.running).toBe(false);
    });
    it('rejects initialization promptly when the backend closes', async () => {
        reply = {type: 'session.closed', diagnostic: {message: 'invalid reference'}};
        const session = new RealtimeSession('test', {});
        await expect(session.start('', {sampling: {}})).rejects.toThrow('invalid reference');
        expect(session.running).toBe(false);
        expect(session.ws.readyState).toBe(3);
    });
});
