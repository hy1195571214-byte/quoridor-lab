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

let audioContext;
export function playMoveSound(muted, isWall = false) {
  if (muted) return;
  try {
    audioContext ??= new AudioContext();
    audioContext.resume().catch(() => {});
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = "triangle";
    oscillator.frequency.setValueAtTime(
      isWall ? 240 : 560,
      audioContext.currentTime,
    );
    oscillator.frequency.exponentialRampToValueAtTime(
      120,
      audioContext.currentTime + 0.08,
    );
    gain.gain.setValueAtTime(0.06, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.001,
      audioContext.currentTime + 0.09,
    );
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.1);
  } catch {
    /* Audio availability does not affect game state. */
  }
}
