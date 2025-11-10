// Minimal Supabase client (single place that imports the npm package).
// Make sure you run:  npm i @supabase/supabase-js
// And set .env values: VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY

import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string;
const anon = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!url || !anon) {
  // Helps you catch misconfigured env quickly in dev
  // (won't crash the app in prod)
  console.warn("[Supabase] Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY");
}

export const supabase = createClient(url || "", anon || "");

// A tiny utility device id used for leaderboard rows (no auth complexity)
export function getDeviceId(): string {
  const k = "arfocus.device.id";
  let id = localStorage.getItem(k);
  if (!id) {
    id = crypto.randomUUID?.() || String(Date.now());
    localStorage.setItem(k, id);
  }
  return id;
}

export function getDisplayName(): string {
  const k = "arfocus.display.name";
  let nm = localStorage.getItem(k);
  if (!nm) {
    nm = "Anonymous";
    localStorage.setItem(k, nm);
  }
  return nm;
}

export function setDisplayName(name: string) {
  localStorage.setItem("arfocus.display.name", name || "Anonymous");
}
