import moveUrl from "../assets/move.mp3";
import wallUrl from "../assets/wall.mp3";

export function readPreference(key, fallback) {
  try {
    const value = localStorage.getItem(`barricade-lab:${key}`);
    return value === null ? fallback : JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function writePreference(key, value) {
  try {
    localStorage.setItem(`barricade-lab:${key}`, JSON.stringify(value));
  } catch {
    /* Browsing remains usable when storage is unavailable. */
  }
}

// Decode once per AudioContext: AudioBuffers are shared across destinations.
const decodedByContext = new WeakMap();
export function preloadGameSounds(context) {
  let promise = decodedByContext.get(context);
  if (!promise) {
    promise = Promise.all([moveUrl, wallUrl].map(async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Sound request failed: ${response.status}`);
      return context.decodeAudioData(await response.arrayBuffer());
    })).catch((error) => {
      decodedByContext.delete(context);
      throw error;
    });
    decodedByContext.set(context, promise);
  }
  return promise;
}

export async function playSoundInto(context, destination, isWall = false) {
  const buffers = await preloadGameSounds(context);
  if (context.state === "closed") return;
  const source = context.createBufferSource();
  source.buffer = buffers[isWall ? 1 : 0];
  source.connect(destination);
  source.start();
}

let audioContext;
export function playMoveSound(muted, isWall = false) {
  if (muted) return;
  try {
    audioContext ??= new AudioContext();
    // Resume synchronously inside the user's gesture before awaiting decoding.
    audioContext.resume().catch(() => {});
    playSoundInto(audioContext, audioContext.destination, isWall).catch(() => {});
  } catch {
    /* Audio availability does not affect game state. */
  }
}
