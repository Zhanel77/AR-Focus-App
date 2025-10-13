import React, { useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";
import * as faceLandmarks from "@tensorflow-models/face-landmarks-detection";
import { Play, Pause, RotateCcw, Camera, ShieldCheck, Info, TrendingUp, Download } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const STORAGE_KEY = "arfocus.sessions.v1";

function formatMMSS(totalSeconds: number) {
  const m = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const s = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function todayKey(d: Date) {
  return d.toISOString().slice(0, 10);
}

type SessionLog = {
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
  mode: "alpha" | "beta";
};

function loadSessions(): SessionLog[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr as SessionLog[];
  } catch {
    return [];
  }
}

function saveSessions(list: SessionLog[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

function toCSV(rows: SessionLog[]) {
  const header = [
    "id","startedAt","endedAt","durationSec","workMinutes","breakMinutes","focusSec","distractSec","hpStart","hpEnd","mode"
  ];
  const body = rows.map(r => [r.id,r.startedAt,r.endedAt,r.durationSec,r.workMinutes,r.breakMinutes,r.focusSec,r.distractSec,r.hpStart,r.hpEnd,r.mode].join(","));
  return [header.join(","), ...body].join("\n");
}

export default function ARFocusMVP() {
  const [workMinutes, setWorkMinutes] = useState(25);
  const [breakMinutes, setBreakMinutes] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [isOnBreak, setIsOnBreak] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(workMinutes * 60);

  const [hp, setHp] = useState(100);
  const [cameraOn, setCameraOn] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  const [inferenceActive, setInferenceActive] = useState(false);
  const [focusedNow, setFocusedNow] = useState<boolean | null>(null);
  const [distractionStreakMs, setDistractionStreakMs] = useState(0);

  const distractGraceMs = 1500;
  const focusRewardPerSec = 0.06;
  const distractPenaltyPerSec = 0.5;

  const [focusSec, setFocusSec] = useState(0);
  const [distractSec, setDistractSec] = useState(0);
  const hpStartRef = useRef(100);

  const [mode, setMode] = useState<"alpha" | "beta">("alpha");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<faceLandmarks.FaceLandmarksDetector | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastInferTs = useRef<number>(0);

  useEffect(() => {
    if (!isRunning && !isOnBreak) setSecondsLeft(workMinutes * 60);
  }, [workMinutes, isRunning, isOnBreak]);

  useEffect(() => {
    if (!isRunning && isOnBreak) setSecondsLeft(breakMinutes * 60);
  }, [breakMinutes, isRunning, isOnBreak]);

  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          const nextIsBreak = !isOnBreak;
          setIsOnBreak(nextIsBreak);
          return (nextIsBreak ? breakMinutes : workMinutes) * 60;
        }
        return s - 1;
      });
      setHp((h) => {
        const delta = focusedNow ? focusRewardPerSec : -distractPenaltyPerSec;
        const nh = Math.max(0, Math.min(100, h + delta));
        return nh;
      });
      if (focusedNow) setFocusSec((x) => x + 1);
      else setDistractSec((x) => x + 1);
    }, 1000);
    return () => clearInterval(id);
  }, [isRunning, isOnBreak, breakMinutes, workMinutes, focusedNow]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    async function boot() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();
        if (cameraOn) {
          stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
          }
        }
        try {
          const detector = await faceLandmarks.createDetector(
            faceLandmarks.SupportedModels.MediaPipeFaceMesh,
            {
              runtime: "tfjs",        // use TensorFlow.js backend (runs fully in browser)
              refineLandmarks: true,  // required by newer versions of the model
              maxFaces: 1             // only track one user for focus detection
            }
          );
          detectorRef.current = detector;
          setModelReady(true);
        } catch (e) {
          console.warn("Face model failed to load:", e);
          setModelReady(false);
        }
        if (!cancelled) setInferenceActive(true);
      } catch (err) {
        console.warn("Camera/TFJS init failed:", err);
        setInferenceActive(false);
      }
    }
    boot();

    return () => {
      cancelled = true;
      setInferenceActive(false);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [cameraOn]);

  useEffect(() => {
    if (!inferenceActive) return;

    const loop = async (ts: number) => {
      const v = videoRef.current;
      const c = canvasRef.current;
      const det = detectorRef.current;

      if (v && c) {
        const ctx = c.getContext("2d");
        if (ctx) {
          c.width = v.videoWidth;
          c.height = v.videoHeight;
          ctx.drawImage(v, 0, 0, c.width, c.height);
        }
      }

      if (det && modelReady && videoRef.current && videoRef.current.readyState >= 2) {
        const elapsed = ts - lastInferTs.current;
        if (elapsed > 200) {
          lastInferTs.current = ts;
          try {
            const faces = await det.estimateFaces(videoRef.current as HTMLVideoElement, { flipHorizontal: true });
            const present = Array.isArray(faces) && faces.length > 0;

            if (present && canvasRef.current) {
              const ctx = canvasRef.current.getContext("2d");
              const f = faces[0] as any;
              if (ctx && f?.box) {
                const { xMin, yMin, width, height } = f.box;
                ctx.strokeStyle = "#22c55e";
                ctx.lineWidth = 2;
                ctx.strokeRect(xMin, yMin, width, height);
              }
            }

            if (!present) {
              setDistractionStreakMs((ms) => {
                const next = ms + elapsed;
                if (next > distractGraceMs) setFocusedNow(false);
                return next;
              });
            } else {
              setDistractionStreakMs(0);
              setFocusedNow(true);
            }
          } catch (e) {
            setFocusedNow(null);
          }
        }
      } else {
        setFocusedNow(null);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [inferenceActive, modelReady, distractGraceMs]);

  const [sessions, setSessions] = useState<SessionLog[]>(() => loadSessions());
  const sessionStartRef = useRef<string | null>(null);

  function startTimer() {
    if (!isRunning) {
      setIsRunning(true);
      if (!sessionStartRef.current) {
        sessionStartRef.current = new Date().toISOString();
        hpStartRef.current = hp;
      }
    }
  }
  function pauseTimer() {
    setIsRunning(false);
  }
  function resetTimer() {
    setIsRunning(false);
    setIsOnBreak(false);
    setSecondsLeft(workMinutes * 60);
    setFocusSec(0);
    setDistractSec(0);
    setHp(100);
    sessionStartRef.current = null;
  }

  function stopAndSave() {
    setIsRunning(false);
    const startedAt = sessionStartRef.current || new Date().toISOString();
    const endedAt = new Date().toISOString();

    const log: SessionLog = {
      id: `${Date.now()}`,
      startedAt,
      endedAt,
      durationSec: focusSec + distractSec,
      workMinutes,
      breakMinutes,
      focusSec,
      distractSec,
      hpStart: hpStartRef.current,
      hpEnd: hp,
      mode,
    };
    const next = [log, ...sessions].slice(0, 300);
    setSessions(next);
    saveSessions(next);

    setIsOnBreak(false);
    setSecondsLeft(workMinutes * 60);
    setFocusSec(0);
    setDistractSec(0);
    setHp(100);
    sessionStartRef.current = null;
  }

  const chartData = useMemo(() => {
    const byDay: Record<string, { date: string; focus: number; distract: number }> = {};
    for (const s of sessions) {
      const key = s.startedAt.slice(0, 10);
      if (!byDay[key]) byDay[key] = { date: key, focus: 0, distract: 0 };
      byDay[key].focus += s.focusSec / 60;
      byDay[key].distract += s.distractSec / 60;
    }
    const keys = Object.keys(byDay).sort().slice(-7);
    return keys.map((k) => byDay[k]);
  }, [sessions]);

  const totalPhaseSec = (isOnBreak ? breakMinutes : workMinutes) * 60;
  const pct = ((totalPhaseSec - secondsLeft) / totalPhaseSec) * 100;

  return (
    <div className="min-h-screen w-full bg-slate-50 text-slate-900">
      <div className="max-w-5xl mx-auto px-4 py-6">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-2xl bg-indigo-100 text-indigo-700"><Camera size={18} /></div>
            <h1 className="text-xl font-semibold">AR Focus App — MVP</h1>
            <span className="ml-2 text-xs px-2 py-1 rounded-full bg-slate-200">{mode.toUpperCase()} TEST</span>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1"><ShieldCheck size={14} /><span>On-device only • No recording</span></div>
          </div>
        </header>

        <section className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="col-span-1 bg-white rounded-2xl shadow p-4 flex flex-col items-center">
            <div className="relative w-44 h-44">
              <svg className="w-44 h-44 -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" stroke="#e5e7eb" strokeWidth="10" fill="none" />
                <circle cx="50" cy="50" r="45" stroke={isOnBreak ? "#f59e0b" : "#6366f1"} strokeWidth="10" fill="none" strokeDasharray={`${(pct/100)*2*Math.PI*45} ${2*Math.PI*45}`} strokeLinecap="round" />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-3xl font-bold tabular-nums">{formatMMSS(secondsLeft)}</div>
                <div className={`text-xs mt-1 ${isOnBreak ? "text-amber-600" : "text-indigo-600"}`}>{isOnBreak ? "Break" : "Focus"}</div>
              </div>
            </div>

            <div className="mt-4 flex gap-2">
              {!isRunning ? (
                <button onClick={startTimer} className="px-3 py-2 rounded-xl bg-indigo-600 text-white flex items-center gap-2 shadow">
                  <Play size={16} /> Start
                </button>
              ) : (
                <button onClick={pauseTimer} className="px-3 py-2 rounded-xl bg-slate-800 text-white flex items-center gap-2 shadow">
                  <Pause size={16} /> Pause
                </button>
              )}
              <button onClick={resetTimer} className="px-3 py-2 rounded-xl bg-slate-200 text-slate-900 flex items-center gap-2">
                <RotateCcw size={16} /> Reset
              </button>
              <button onClick={stopAndSave} className="px-3 py-2 rounded-xl bg-emerald-600 text-white flex items-center gap-2">
                <TrendingUp size={16} /> Save
              </button>
            </div>

            <div className="mt-3 text-xs text-slate-500 flex items-center gap-2">
              <Info size={14} /> Focus adds HP slowly; distractions reduce it faster.
            </div>
          </div>

          <div className="col-span-1 bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-3">Session Settings</h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm">Work (min)</label>
                <input type="number" min={5} max={120} value={workMinutes}
                  onChange={(e) => setWorkMinutes(Math.max(5, Math.min(120, Number(e.target.value))))}
                  className="w-24 px-2 py-1 rounded-xl bg-slate-100" />
              </div>
              <div className="flex items-center justify-between">
                <label className="text-sm">Break (min)</label>
                <input type="number" min={3} max={60} value={breakMinutes}
                  onChange={(e) => setBreakMinutes(Math.max(3, Math.min(60, Number(e.target.value))))}
                  className="w-24 px-2 py-1 rounded-xl bg-slate-100" />
              </div>
              <div className="flex items-center justify-between">
                <label className="text-sm">Mode</label>
                <select value={mode} onChange={(e) => setMode(e.target.value as any)} className="w-28 px-2 py-1 rounded-xl bg-slate-100">
                  <option value="alpha">Alpha</option>
                  <option value="beta">Beta</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <label className="text-sm">Camera</label>
                <button onClick={() => setCameraOn((v) => !v)} className={`px-3 py-1 rounded-xl ${cameraOn ? "bg-emerald-600 text-white" : "bg-slate-200"}`}>
                  {cameraOn ? "On" : "Off"}
                </button>
              </div>
              <div className="text-xs text-slate-500 leading-relaxed">
                • All processing is local in your browser. • No video is recorded or uploaded.
              </div>
            </div>
          </div>

          <div className="col-span-1 bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-3">Live Status</h2>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="p-3 rounded-xl bg-slate-100 flex flex-col">
                <span className="text-slate-500">HP</span>
                <span className="text-xl font-semibold">{hp.toFixed(1)}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-100 flex flex-col">
                <span className="text-slate-500">Attention</span>
                <span className={`text-sm font-semibold ${focusedNow===true?"text-emerald-600":focusedNow===false?"text-rose-600":"text-slate-500"}`}>
                  {focusedNow === true ? "Focused" : focusedNow === false ? "Distracted" : "Unknown"}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-100 flex flex-col">
                <span className="text-slate-500">Focused (s)</span>
                <span className="text-xl font-semibold">{focusSec}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-100 flex flex-col">
                <span className="text-slate-500">Distracted (s)</span>
                <span className="text-xl font-semibold">{distractSec}</span>
              </div>
            </div>
            <div className="mt-3 text-xs text-slate-500">
              {modelReady ? "Face model active" : "Face model not loaded (timer still works)"}
            </div>
          </div>
        </section>

        <section className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-black rounded-2xl overflow-hidden shadow relative">
            <video ref={videoRef} playsInline className="w-full h-[260px] object-cover opacity-70" muted />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-[260px]" />
            {!cameraOn && (
              <div className="absolute inset-0 flex items-center justify-center text-white/80 text-sm">Camera is off</div>
            )}
          </div>

          <div className="bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-3 flex items-center gap-2"><TrendingUp size={18}/> Weekly Focus vs Distraction (min)</h2>
            <div className="w-full h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="focus" stroke="#22c55e" strokeWidth={2} dot={false} name="Focus" />
                  <Line type="monotone" dataKey="distract" stroke="#ef4444" strokeWidth={2} dot={false} name="Distract" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        <section className="mt-6 bg-white rounded-2xl shadow p-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Saved Sessions</h2>
            <button
              onClick={() => {
                const csv = toCSV(sessions);
                const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `arfocus_sessions_${Date.now()}.csv`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="px-3 py-2 rounded-xl bg-slate-800 text-white flex items-center gap-2"
            >
              <Download size={16} /> Export CSV
            </button>
          </div>

          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-slate-500">
                  <th className="py-2 pr-4">Date</th>
                  <th className="py-2 pr-4">Dur</th>
                  <th className="py-2 pr-4">Focus</th>
                  <th className="py-2 pr-4">Distract</th>
                  <th className="py-2 pr-4">HP Δ</th>
                  <th className="py-2 pr-4">Mode</th>
                </tr>
              </thead>
              <tbody>
                {sessions.length === 0 && (
                  <tr><td className="py-3 text-slate-500" colSpan={6}>No sessions yet — press Start, then Save.</td></tr>
                )}
                {sessions.map((s) => (
                  <tr key={s.id} className="border-t border-slate-100">
                    <td className="py-2 pr-4">{new Date(s.startedAt).toLocaleString()}</td>
                    <td className="py-2 pr-4">{formatMMSS(s.durationSec)}</td>
                    <td className="py-2 pr-4">{formatMMSS(s.focusSec)}</td>
                    <td className="py-2 pr-4">{formatMMSS(s.distractSec)}</td>
                    <td className="py-2 pr-4">{(s.hpEnd - s.hpStart).toFixed(1)}</td>
                    <td className="py-2 pr-4 uppercase">{s.mode}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <footer className="mt-8 text-xs text-slate-500 leading-relaxed">
          This MVP checks for face presence to infer attention. It does not store video and works offline after model load. For improved accuracy (gaze/phone detection), iterate in future sprints.
        </footer>
      </div>
    </div>
  );
}
