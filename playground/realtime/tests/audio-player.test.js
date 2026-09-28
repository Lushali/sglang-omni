import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { AudioPlayer } from '../frontend/shared/duplex/lib/audio-player.js';

let player;
let metrics;
let sources;
beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.stubGlobal('requestAnimationFrame', callback => callback());
    sources = [];
    vi.stubGlobal('AudioContext', class {
        sampleRate = 100;
        state = 'running';
        started = Date.now();
        get currentTime() { return (Date.now() - this.started) / 1000; }
        createBuffer(channels, length, rate) {
            return {duration: length / rate, getChannelData: () => new Float32Array(length)};
        }
        createBufferSource() {
            const source = {connect() {}, disconnect() {}, start: vi.fn(), stop: vi.fn()};
            sources.push(source);
            return source;
        }
    });
    metrics = [];
    player = new AudioPlayer({outputSampleRate: 100, getPlaybackDelayMs: () => 0});
    player.onMetrics = value => metrics.push(value);
    player.init();
    player.beginTurn();
    const samples = new Float32Array(200);
    player.playChunk(btoa(String.fromCharCode(...new Uint8Array(samples.buffer))));
});
afterEach(() => {
    player.stop();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

it('keeps reporting the remaining audio after the response ends until it drains', () => {
    player.endTurn();
    vi.advanceTimersByTime(1000);
    expect(metrics.at(-1).ahead).toBeCloseTo(1000);
    expect(sources[0].stop).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1200);
    expect(metrics.at(-1).ahead).toBe(0);
    expect(player.lastAheadMs).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
});

it('reports the current remaining audio between metric ticks for session cleanup', () => {
    player.endTurn();
    vi.advanceTimersByTime(50);
    expect(player.lastAheadMs).toBeCloseTo(1950);
});

it('clears the displayed and remaining audio immediately when playback is stopped', () => {
    player.stopAll();
    expect(sources[0].stop).toHaveBeenCalledOnce();
    expect(player.lastAheadMs).toBe(0);
    expect(metrics.at(-1).ahead).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
});
