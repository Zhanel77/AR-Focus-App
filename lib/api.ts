// Supabase-backed APIs (sessions + leaderboard + optional challenges)

import { supabase, getDeviceId, getDisplayName } from "../lib/supabaseClient";

// Duplicate of your SessionLog to avoid circular imports
export type SessionLog = {
  id: string;
  startedAt: string;
  endedAt: string;
  durationSec: number;
  workMinutes: number;
  breakMinutes: number;
  focusSec: number;
  distractSec: number;
  hpStart: number;
  hpEnd: number;
};

export type LeaderboardRow = {
  device_id: string;
  display_name: string | null;
  score: number;
  updated_at: string;
};

export function scoreFromSession(s: SessionLog): number {
  // Simple, transparent scoring you can tweak later
  // Focus seconds minus a fraction of distraction
  const base = Math.max(0, Math.round(s.focusSec - 0.5 * s.distractSec));
  const bonus = Math.max(0, Math.round((s.hpEnd - s.hpStart) * 2));
  return Math.max(1, base + bonus);
}

// Push a saved session + upsert leaderboard row
export async function postSessionToCloud(s: SessionLog) {
  if (!supabase) return;

  const device_id = getDeviceId();
  const display_name = getDisplayName();
  const score = scoreFromSession(s);

  // 1) store raw session (table: sessions)
  await supabase.from("sessions").insert({
    device_id,
    started_at: s.startedAt,
    ended_at: s.endedAt,
    duration_sec: s.durationSec,
    focus_sec: s.focusSec,
    distract_sec: s.distractSec,
    hp_start: s.hpStart,
    hp_end: s.hpEnd,
  });

  // 2) update leaderboard (table: leaderboard, unique by device_id)
  await supabase
    .from("leaderboard")
    .upsert(
      { device_id, display_name, score, updated_at: new Date().toISOString() },
      { onConflict: "device_id" }
    );
}

export async function fetchLeaderboard(limit = 10): Promise<LeaderboardRow[]> {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from("leaderboard")
    .select("*")
    .order("score", { ascending: false })
    .limit(limit);
  if (error) {
    console.warn("[leaderboard] fetch error:", error.message);
    return [];
  }
  return (data || []) as LeaderboardRow[];
}

// --- Optional: simple challenges (tables: challenges, challenge_members) ---

export type Challenge = {
  id: string;
  title: string;
  starts_at: string | null;
  ends_at: string | null;
  is_public: boolean | null;
};

export async function fetchActiveChallenges(): Promise<Challenge[]> {
  const now = new Date().toISOString();
  const { data, error } = await supabase
    .from("challenges")
    .select("*")
    .lte("starts_at", now)
    .gte("ends_at", now)
    .order("starts_at", { ascending: true });
  if (error) {
    console.warn("[challenges] fetch error:", error.message);
    return [];
  }
  return (data || []) as Challenge[];
}

export async function joinChallenge(challengeId: string) {
  const device_id = getDeviceId();
  const display_name = getDisplayName();
  const { error } = await supabase
    .from("challenge_members")
    .upsert({ challenge_id: challengeId, device_id, display_name });
  if (error) console.warn("[challenges] join error:", error.message);
}

export async function postChallengeScore(challengeId: string, score: number) {
  const device_id = getDeviceId();
  const { error } = await supabase
    .from("challenge_members")
    .upsert({ challenge_id: challengeId, device_id, score, updated_at: new Date().toISOString() });
  if (error) console.warn("[challenges] score error:", error.message);
}
