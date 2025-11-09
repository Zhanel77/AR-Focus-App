import { read, write } from "../lib/storage";

export type Theme = "angelic" | "demonic";
export type Difficulty = "easy" | "normal" | "hard";

export type Settings = {
  theme: Theme;
  difficulty: Difficulty;
  workMin: number;
  breakMin: number;
  tipsOn: boolean;
};

const KEY = "arfocus.settings.v1";

let state: Settings = read<Settings>(KEY, {
  theme: "demonic",
  difficulty: "normal",
  workMin: 25,
  breakMin: 5,
  tipsOn: true,
});

const listeners = new Set<() => void>();
export function useSettings(): [Settings, (patch: Partial<Settings>) => void] {
  // tiny “store” with manual subscribe (no libs)
  const set = (patch: Partial<Settings>) => {
    state = { ...state, ...patch };
    write(KEY, state);
    listeners.forEach(fn => fn());
  };
  // naive hook-less consumer: caller subscribes in useEffect
  // For now you can import getSettings() directly instead of a hook.
  return [state, set] as any;
}
export function getSettings() { return state; }
export function subscribeSettings(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

