// pages/Leaderboard.tsx
import React, { useEffect, useState } from "react";
import { supabase, getDeviceId } from "../lib/supabaseClient";
import { fetchLeaderboard } from "../lib/api";

type SessionRow = { created_at: string; focus_sec: number; distract_sec: number };
type LBRow = { device_id: string; display_name: string | null; score: number; updated_at: string };

export default function Leaderboard({
  onClose,
  isDark = false,
}: {
  onClose: () => void;
  isDark?: boolean;
}) {
  const [tab, setTab] = useState<"global" | "sessions">("global");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const [topSessions, setTopSessions] = useState<
    { rate: number; total: number; date: string }[]
  >([]);
  const [lbRows, setLbRows] = useState<LBRow[]>([]);
  const myId = getDeviceId();

  // ---- theme helpers (match App)
  const pane   = isDark ? "bg-slate-900/90 text-slate-100" : "bg-white text-slate-900";
  const soft   = isDark ? "bg-slate-800/60"                 : "bg-slate-100";
  const subtle = isDark ? "text-slate-400"                  : "text-slate-600";
  const accent = isDark ? "text-indigo-300"                 : "text-indigo-600";
  const rowBorder = isDark ? "border-slate-800" : "border-slate-100";
  const tabBase = "px-3 py-1 rounded-xl text-sm transition";
  const tabActive = isDark ? "bg-indigo-600/30 text-indigo-200" : "bg-slate-900 text-white";
  const tabIdle = soft;

  async function loadSessions() {
    setLoading(true); setErr("");
    const { data, error } = await supabase
      .from("sessions")
      .select("created_at, focus_sec, distract_sec")
      .order("created_at", { ascending: false })
      .limit(10);

    if (error) {
      setErr(error.message);
      setTopSessions([]);
    } else {
      const mapped = (data as SessionRow[]).map((r) => {
        const f = r.focus_sec ?? 0, d = r.distract_sec ?? 0;
        const total = Math.max(1, f + d);
        const rate = f / total;
        return { rate, total, date: r.created_at };
      });
      setTopSessions(mapped);
    }
    setLoading(false);
  }

  async function loadLeaderboard() {
    setLoading(true); setErr("");
    const data = await fetchLeaderboard(20);
    setLbRows(data);
    setLoading(false);
  }

  useEffect(() => {
    if (tab === "sessions") loadSessions();
    else loadLeaderboard();
  }, [tab]);

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className={`${pane} rounded-2xl shadow-xl max-w-2xl w-full p-6 relative`}>
        <button
          className={`absolute top-3 right-3 ${subtle} hover:!text-slate-300`}
          onClick={onClose}
          aria-label="Close leaderboard"
          title="Close"
        >
          ✕
        </button>

        <div className="flex items-center justify-between mb-3">
          <h2 className={`text-lg font-semibold ${accent}`}>
            {tab === "global" ? "Leaderboard (Best per Device)" : "Top Sessions"}
          </h2>
          <div className="flex gap-2">
            <button
              onClick={() => (tab === "global" ? loadLeaderboard() : loadSessions())}
              className={`text-xs px-2 py-1 rounded-lg ${soft}`}
              disabled={loading}
            >
              {loading ? "…" : "Refresh"}
            </button>
          </div>
        </div>

        <div className="mb-3 flex gap-2">
          <button
            className={`${tabBase} ${tab === "global" ? tabActive : tabIdle}`}
            onClick={() => setTab("global")}
          >
            Leaderboard
          </button>
          <button
            className={`${tabBase} ${tab === "sessions" ? tabActive : tabIdle}`}
            onClick={() => setTab("sessions")}
          >
            Top Sessions
          </button>
        </div>

        {err && <div className={`text-xs text-rose-500 mb-2`}>Error: {err}</div>}

        {/* ------ GLOBAL LEADERBOARD (best per device) ------ */}
        {tab === "global" && (
          <div className="overflow-x-auto text-sm">
            <table className="min-w-full">
              <thead>
                <tr className={`text-left ${subtle}`}>
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">User</th>
                  <th className="py-2 pr-3">Score</th>
                  <th className="py-2 pr-3">Updated</th>
                </tr>
              </thead>
              <tbody>
                {lbRows.length === 0 && !loading && (
                  <tr>
                    <td className={`py-3 ${subtle}`} colSpan={4}>No data yet.</td>
                  </tr>
                )}
                {lbRows.map((r, i) => {
                  const isMe = r.device_id === myId;
                  return (
                    <tr key={r.device_id} className={`border-t ${rowBorder}`}>
                      <td className="py-1 pr-3">{i + 1}</td>
                      <td className="py-1 pr-3">
                        {isMe ? "⭐ " : ""}
                        {r.display_name || "Anonymous"}
                      </td>
                      <td className="py-1 pr-3">{r.score}</td>
                      <td className="py-1 pr-3">
                        {new Date(r.updated_at).toLocaleDateString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* ------ TOP SESSIONS (recent runs) ------ */}
        {tab === "sessions" && (
          <div className="overflow-x-auto text-sm">
            <table className="min-w-full">
              <thead>
                <tr className={`text-left ${subtle}`}>
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">Focus %</th>
                  <th className="py-2 pr-3">Duration (s)</th>
                  <th className="py-2 pr-3">Date</th>
                </tr>
              </thead>
              <tbody>
                {topSessions.length === 0 && !loading && (
                  <tr>
                    <td className={`py-3 ${subtle}`} colSpan={4}>No data yet.</td>
                  </tr>
                )}
                {topSessions.map((r, i) => (
                  <tr key={i} className={`border-t ${rowBorder}`}>
                    <td className="py-1 pr-3">{i + 1}</td>
                    <td className="py-1 pr-3">{Math.round(r.rate * 100)}%</td>
                    <td className="py-1 pr-3">{Math.round(r.total)}</td>
                    <td className="py-1 pr-3">{new Date(r.date).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
