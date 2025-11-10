// ==========================================================================
// 🧩 PART 1 — Supabase + Free Exam Season + User State + Motivation
// ==========================================================================

// --- 1️⃣  Imports ----------------------------------------
import { supabase } from "../lib/supabaseClient";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { postSessionToCloud } from "../lib/api";
import Leaderboard from "../pages/Leaderboard";

// --- 3️⃣  Free Exam Season Logic ------------------------------------------
export function isFreeExamSeason(): boolean {
  const m = new Date().getMonth() + 1;
  return m === 5 || m === 12 || localStorage.getItem("forceExamUnlock") === "true";
}
export async function handleExamSeasonUnlock() {
  const active = isFreeExamSeason();
  if (active) {
    console.log("🎓 Free Exam Season active — Premium features unlocked");
    localStorage.setItem("premiumUnlocked", "true");
  } else {
    localStorage.removeItem("premiumUnlocked");
  }
  return active;
}
export function useExamSeasonBoot() {
  useEffect(() => {
    handleExamSeasonUnlock().catch((e) => console.warn("Exam season check failed:", e));
  }, []);
}

// --- 4️⃣  User Profile Hooks ----------------------------------------------
export interface UserProfile {
  id: string;
  name: string;
  email?: string;
  theme: "light" | "dark";
  difficulty: "easy" | "normal" | "hard";
  premiumUnlocked: boolean;
}
export function useUserProfile() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  useEffect(() => {
    const stored = localStorage.getItem("arfocus.user");
    if (stored) setProfile(JSON.parse(stored));
    else {
      const anon = {
        id: crypto.randomUUID(),
        name: "Guest",
        theme: "light",
        difficulty: "normal",
        premiumUnlocked: localStorage.getItem("premiumUnlocked") === "true",
      } as UserProfile;
      localStorage.setItem("arfocus.user", JSON.stringify(anon));
      setProfile(anon);
    }
  }, []);
  function updateProfile(partial: Partial<UserProfile>) {
    if (!profile) return;
    const next = { ...profile, ...partial };
    setProfile(next);
    localStorage.setItem("arfocus.user", JSON.stringify(next));
  }
  return { profile, updateProfile };
}

// --- 6️⃣  Daily Motivation Tips ------------------------------------------
const MOTIVATION_QUOTES = [
  "Small steps lead to big progress.",
  "Focus is a muscle — train it daily.",
  "Your future self will thank you for today’s discipline.",
  "Consistency beats intensity every time.",
  "You don’t need motivation when you have momentum.",
  "Great work comes from great focus — stay present.",
  "Every distraction you resist is a win for your goals.",
  "The more you practice focus, the easier it becomes.",
];
export function useDailyMotivation() {
  const [quote, setQuote] = useState("");
  useEffect(() => {
    const idx = Math.floor(Math.random() * MOTIVATION_QUOTES.length);
    setQuote(MOTIVATION_QUOTES[idx]);
  }, []);
  return quote;
}

// --- 7️⃣  Smart Notification Placeholder ----------------------------------
export function useSmartNotifications(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;
    console.log("🔕 Smart Notification Filter enabled");
    const prev = Notification.permission;
    if (prev === "default") Notification.requestPermission();
    const timer = setInterval(() => {
      console.log("Filtering distracting apps notifications …");
    }, 60000);
    return () => clearInterval(timer);
  }, [enabled]);
}
import { X } from "lucide-react";
import {
  FilesetResolver,
  FaceLandmarker,
  ObjectDetector,
} from "@mediapipe/tasks-vision";
import {
  Play,
  Pause,
  RotateCcw,
  Camera,
  ShieldCheck,
  Info,
  TrendingUp,
  Download,
  Volume2,
  VolumeX,
  Smartphone,
} from "lucide-react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ----------------------------- Tweakables ---------------------------------
const STORAGE_KEY = "arfocus.sessions.v1";

const INFER_HZ = 10;
const EMA_ALPHA = 0.25;
const BG_INTERVAL_MS = 400;

const PHONE_DETECT_EVERY = 2;       // run detector every N frames

const NO_FACE_GRACE_MS = 1500;      // buffer before "no face" soft logic
const NO_FACE_STRONG_MS = 2000;     // ≥2s face missing/camera dark => ≥Strong
const EYES_AWAY_MIN_MS = 1000;      // Mild if gaze away >1s
const EYES_NOT_VISIBLE_MS = 2000;   // Strong if eyes not visible >2s
const EYE_CLOSED_MAX_MS = 1000;     // ≤1s closed/covered ⇒ distraction
const EYE_OPEN_THRESH = 0.18;

// ---- Beep settings (continuous tone) ----
const BEEP_START_DELAY_MS = 600;   // wait this long after distraction before tone
const BEEP_FREQ_HZ = 880;          // pitch of the tone (Hz)
const BEEP_MAX_GAIN = 0.25;        // safety cap; final volume = slider * this

// Models
const MP_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MP_FACE_TASK =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const MP_OBJ_TASK =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.task";
const MP_OBJ_TASK_FALLBACK =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";

// Productive tabs heartbeat channel
const PRODUCTIVE_CHANNEL = "arfocus-productive";
const PRODUCTIVE_PING_FRESH_MS = 2500; // recent ping = productive even if hidden

// ----------------------------- TS Helpers ---------------------------------
// Make ImageCapture safe for TS environments that lack the DOM type
declare global {
  interface Window {
    ImageCapture?: any;
  }
}
type AnyImageCapture = any;

// ----------------------------- Utils --------------------------------------
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
};

function loadSessions(): SessionLog[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? (arr as SessionLog[]) : [];
  } catch {
    return [];
  }
}
function saveSessions(list: SessionLog[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

function formatMMSS(totalSeconds: number) {
  const m = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const s = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
function mmss(totalSec: number) {
  const m = Math.floor(totalSec / 60).toString().padStart(2, "0");
  const s = Math.floor(totalSec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
function formatLocal(dtIso: string) {
  const d = new Date(dtIso);
  return d.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
function fmtPct(n: number) {
  if (!isFinite(n)) return "—";
  return `${(n * 100).toFixed(0)}%`;
}
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const now = () => performance.now();

function toCSV(rows: SessionLog[]) {
  const header = [
    "id",
    "startedAt",
    "endedAt",
    "durationSec",
    "workMinutes",
    "breakMinutes",
    "focusSec",
    "distractSec",
    "hpStart",
    "hpEnd",
  ];
  const body = rows.map((r) =>
    [
      r.id,
      r.startedAt,
      r.endedAt,
      r.durationSec,
      r.workMinutes,
      r.breakMinutes,
      r.focusSec,
      r.distractSec,
      r.hpStart,
      r.hpEnd,
    ].join(",")
  );
  return [header.join(","), ...body].join("\n");
}

// Euler from MediaPipe 4x4 (column-major Float32Array length 16)
function eulerFromMat4Deg(mIn: Float32Array | number[]) {
  const m = mIn instanceof Float32Array ? mIn : new Float32Array(mIn);
  const r00 = m[0],  r01 = m[4],  r02 = m[8];
  const r10 = m[1],  r11 = m[5],  r12 = m[9];
  const r20 = m[2],  r21 = m[6],  r22 = m[10];
  const pitch = Math.atan2(-r21, r22); // X
  const roll  = Math.atan2(-r01, r00); // Z
  const yaw   = Math.asin(Math.max(-1, Math.min(1, r20))); // Y
  const rad2deg = 180 / Math.PI;
  return { pitchDeg: pitch * rad2deg, yawDeg: yaw * rad2deg, rollDeg: roll * rad2deg };
}

function isVeryDark(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const sample = ctx.getImageData(
    0, 0,
    Math.max(1, Math.floor(w / 8)),
    Math.max(1, Math.floor(h / 8))
  );
  let s = 0;
  for (let i = 0; i < sample.data.length; i += 4) {
    s += 0.299 * sample.data[i] + 0.587 * sample.data[i + 1] + 0.114 * sample.data[i + 2];
  }
  const avg = s / (sample.data.length / 4);
  return avg < 6;
}

type DistractionLevel = "Focused" | "Mild" | "Strong" | "Critical";

// ----------------------------- Component -----------------------------------
export default function App() {
  useExamSeasonBoot();
  const { profile, updateProfile } = useUserProfile();
  type Difficulty = "easy" | "normal" | "hard";

  const DIFF_PRESETS: Record<Difficulty, {
    ENTER_TH: number;        // EMA gate: enter Focus
    EXIT_TH: number;         // EMA gate: leave Focus (hysteresis)
    CENTER_TOL: number;      // how centered face must be (fraction of width)
    GAZE_TOL: number;        // how centered iris must be (lower=stricter)
    PHONE_TH: number;        // min score to count phone
    PHONE_TH_BOTTOM: number; // same but near bottom of frame
    DOWN_MILD_MS: number;    // look-down mild threshold
    DOWN_STRONG_MS: number;  // look-down strong threshold
    FOCUS_HP_PER_SEC: number;
    DISTRACT_HP_PER_SEC: number;
    defaultWork: number;
    defaultBreak: number;
  }> = {
    easy:   { ENTER_TH: 0.60, EXIT_TH: 0.45, CENTER_TOL: 0.35, GAZE_TOL: 0.38,
              PHONE_TH: 0.55, PHONE_TH_BOTTOM: 0.45, DOWN_MILD_MS: 2000, DOWN_STRONG_MS: 3500,
              FOCUS_HP_PER_SEC: 0.08, DISTRACT_HP_PER_SEC: 0.40, defaultWork: 20, defaultBreak: 5 },
    normal: { ENTER_TH: 0.70, EXIT_TH: 0.50, CENTER_TOL: 0.30, GAZE_TOL: 0.30,
              PHONE_TH: 0.45, PHONE_TH_BOTTOM: 0.35, DOWN_MILD_MS: 1500, DOWN_STRONG_MS: 3000,
              FOCUS_HP_PER_SEC: 0.06, DISTRACT_HP_PER_SEC: 0.50, defaultWork: 25, defaultBreak: 5 },
    hard:   { ENTER_TH: 0.80, EXIT_TH: 0.65, CENTER_TOL: 0.22, GAZE_TOL: 0.22,
              PHONE_TH: 0.35, PHONE_TH_BOTTOM: 0.28, DOWN_MILD_MS: 1200, DOWN_STRONG_MS: 2200,
              FOCUS_HP_PER_SEC: 0.05, DISTRACT_HP_PER_SEC: 0.60, defaultWork: 45, defaultBreak: 5 },
  };

  const activeDiff = (profile?.difficulty ?? "normal") as Difficulty;
  const cfg = React.useMemo(() => DIFF_PRESETS[activeDiff], [activeDiff]);
  const cfgRef = React.useRef(cfg);
  React.useEffect(() => { cfgRef.current = cfg; }, [cfg]);
  const motivation = useDailyMotivation();
  useSmartNotifications(profile?.premiumUnlocked ?? false);
  const [cameraOn, setCameraOn] = useState(true);
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [showCustomize, setShowCustomize] = useState(false);
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [camAR, setCamAR] = useState(4 / 3); // default 640x480
  const [mirrorCam, setMirrorCam] = useState(true);

  // Timer
  const [workMinutes, setWorkMinutes] = useState(25);
  const [breakMinutes, setBreakMinutes] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [isOnBreak, setIsOnBreak] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(workMinutes * 60);

  // HP & sessions
  const [hp, setHp] = useState(100);
  const hpStartRef = useRef(100);
  const [sessions, setSessions] = useState<SessionLog[]>(() => loadSessions());
  const sessionStartRef = useRef<string | null>(null);

  const [focusSec, setFocusSec] = useState(0);
  const [distractSec, setDistractSec] = useState(0);

  // Audio
  const [beepOnDistract, setBeepOnDistract] = useState(true);
  const [beepVolume, setBeepVolume] = useState(0.6);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Web Audio for continuous beep
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscRef = useRef<OscillatorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const beepStartTimerRef = useRef<number | null>(null);


  // Camera & canvases
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // MediaPipe instances
  const faceVideoRef = useRef<FaceLandmarker | null>(null);
  const faceImageRef = useRef<FaceLandmarker | null>(null);
  const objVideoRef = useRef<ObjectDetector | null>(null);
  const objImageRef = useRef<ObjectDetector | null>(null);

  // Background frames
  const imageCaptureRef = useRef<AnyImageCapture | null>(null);
  const rafRef = useRef<number | null>(null);
  const bgIntervalRef = useRef<number | null>(null);

  // Model ready
  const [modelReady, setModelReady] = useState(false);
  const [objectModelReady, setObjectModelReady] = useState(false);

  // Diagnostics & raw signals
  const [facesCount, setFacesCount] = useState(0);
  const [lastError, setLastError] = useState<string>("");

  const [tabHidden, setTabHidden] = useState(false);
  const [gazeOn, setGazeOn] = useState<boolean | null>(null);
  const [eyesVisible, setEyesVisible] = useState(true);
  const [headYaw, setHeadYaw] = useState(0);
  const [headPitch, setHeadPitch] = useState(0);
  const [lookingDown, setLookingDown] = useState(false);
  const [phoneNow, setPhoneNow] = useState(false);
  const [camCovered, setCamCovered] = useState(false);

  // Classifier state
  const [level, setLevel] = useState<DistractionLevel>("Focused");
  // 🔄 Sync updates to the mini dashboard (focus status + timer)
  useEffect(() => {
    const ch = new BroadcastChannel("arfocus-productive");
    ch.postMessage({
      type: "focus:update",
      focused: level === "Focused",
      timer: `${String(Math.floor(secondsLeft / 60)).padStart(2, "0")}:${String(secondsLeft % 60).padStart(2, "0")}`,
    });
    return () => ch.close();
  }, [level, secondsLeft]);
  // 🔄 Listen for control messages from the mini dashboard
  useEffect(() => {
    const ch = new BroadcastChannel("arfocus-productive");

    ch.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg?.type) return;

      if (msg.type === "control:start") startTimer();
      if (msg.type === "control:pause") pauseTimer();
      if (msg.type === "control:reset") resetTimer();
      if (msg.type === "control:save") stopAndSave();
    };

    return () => ch.close();
  }, []);

  // Eyes/phone/tab timers & beep state
  const eyesClosedMsRef = useRef(0);
  const lastPhoneSeenAtRef = useRef(0);
  // const lastBeepAtRef = useRef(0);
  // const prevEffectiveRef = useRef<boolean | null>(null);

  // Productive tabs heartbeat
  const productiveChannelRef = useRef<BroadcastChannel | null>(null);
  const lastProductivePingRef = useRef<number>(-1);
  // Allow working in other tabs without penalty
  const [allowHiddenWork, setAllowHiddenWork] = useState(true);
  useEffect(() => {
    const v = localStorage.getItem("arfocus.allowHiddenWork");
    if (v !== null) setAllowHiddenWork(v === "true");
  }, []);
  useEffect(() => {
    localStorage.setItem("arfocus.allowHiddenWork", String(allowHiddenWork));
  }, [allowHiddenWork]);

  // Track last time vision pipeline ran (even while hidden)
  const lastVisionAtRef = useRef<number>(performance.now());

  // Optional companion window that sends productive pings
  const companionRef = useRef<Window | null>(null);

  // ---- Visibility + productive channel
  useEffect(() => {
    const onVis = () => setTabHidden(document.hidden);
    onVis();
    document.addEventListener("visibilitychange", onVis);
    try {
      const ch = new BroadcastChannel(PRODUCTIVE_CHANNEL);
      productiveChannelRef.current = ch;
      ch.onmessage = (ev) => {
        if (ev?.data?.type === "productive:ping") {
          lastProductivePingRef.current = performance.now();
        }
      };
    } catch {
      productiveChannelRef.current = null;
    }
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      try { productiveChannelRef.current?.close(); } catch {}
    };
  }, []);

  // ---- Pause reset fix (only when IDLE, not when paused) --------------------
  useEffect(() => {
    // Idle + on WORK phase → keep MM:SS synced to workMinutes
    if (!isRunning && !isOnBreak && sessionStartRef.current === null) {
      setSecondsLeft(workMinutes * 60);
    }
  }, [workMinutes, isRunning, isOnBreak]);

  useEffect(() => {
    // Idle + on BREAK phase → keep MM:SS synced to breakMinutes
    if (!isRunning && isOnBreak && sessionStartRef.current === null) {
      setSecondsLeft(breakMinutes * 60);
    }
  }, [breakMinutes, isRunning, isOnBreak]);

  // When difficulty presets change, apply ONLY while truly idle (not paused)
  useEffect(() => {
    // If a session has ever started, we're either running or paused → do nothing
    if (sessionStartRef.current !== null) return;

    if (!isRunning && !isOnBreak) setWorkMinutes(cfg.defaultWork);
    if (!isRunning &&  isOnBreak) setBreakMinutes(cfg.defaultBreak);

    if (!isRunning) {
      setSecondsLeft((isOnBreak ? cfg.defaultBreak : cfg.defaultWork) * 60);
    }
  }, [cfg.defaultWork, cfg.defaultBreak, isRunning, isOnBreak]);



  // ---- Timer tick + HP + stat accumulation (uses classification)
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          const next = !isOnBreak;
          setIsOnBreak(next);
          return (next ? breakMinutes : workMinutes) * 60;
        }
        return s - 1;
      });

      const isFocusedNow = level === "Focused" && !isOnBreak;

      setHp((h) => {
        const { FOCUS_HP_PER_SEC, DISTRACT_HP_PER_SEC } = cfgRef.current;
        const delta = isFocusedNow ? FOCUS_HP_PER_SEC : -DISTRACT_HP_PER_SEC;
        return clamp(h + delta, 0, 100);
      });
      

      if (isFocusedNow) setFocusSec((x) => x + 1);
      else setDistractSec((x) => x + 1);
    }, 1000);
    return () => clearInterval(id);
  }, [isRunning, isOnBreak, workMinutes, breakMinutes, level]);

  // ---- Beep: delayed continuous tone while distracted ----
  useEffect(() => {
    const shouldBeep = beepOnDistract && isRunning && !isOnBreak && level !== "Focused";

    if (shouldBeep) {
      // Start after a small delay to avoid chirps on brief dips
      if (!oscRef.current && beepStartTimerRef.current == null) {
        beepStartTimerRef.current = window.setTimeout(() => {
          beepStartTimerRef.current = null;
          startContinuousBeep();
        }, BEEP_START_DELAY_MS) as unknown as number;
      }
    } else {
      // back to focus, paused, on break, or toggle off
      stopContinuousBeep();
    }
  }, [beepOnDistract, isRunning, isOnBreak, level]);

  // Keep volume responsive to slider
  useEffect(() => {
    if (gainRef.current && audioCtxRef.current) {
      const ctx = audioCtxRef.current;
      const vol = clamp(beepVolume, 0, 1) * BEEP_MAX_GAIN;
      // smooth ramp to avoid clicks
      gainRef.current.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.05);
    }
  }, [beepVolume]);

  // Safety: stop tone if component unmounts
  useEffect(() => () => stopContinuousBeep(), []);

  function startContinuousBeep() {
    // lazy create AudioContext
    if (!audioCtxRef.current) {
      const Ctx: any = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!Ctx) return; // no Web Audio available
      audioCtxRef.current = new Ctx();
    }
    const ctx = audioCtxRef.current!;
    if (ctx.state === "suspended") { try { ctx.resume(); } catch {} }
  
    // already running?
    if (oscRef.current) return;
  
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = BEEP_FREQ_HZ;
  
    const gain = ctx.createGain();
    const vol = clamp(beepVolume, 0, 1) * BEEP_MAX_GAIN;
    gain.gain.setValueAtTime(vol, ctx.currentTime);
  
    osc.connect(gain).connect(ctx.destination);
    osc.start();
  
    oscRef.current = osc;
    gainRef.current = gain;
  }
  
  function stopContinuousBeep() {
    if (beepStartTimerRef.current) {
      window.clearTimeout(beepStartTimerRef.current);
      beepStartTimerRef.current = null;
    }
    try { oscRef.current?.stop(); } catch {}
    try { oscRef.current?.disconnect(); } catch {}
    try { gainRef.current?.disconnect(); } catch {}
    oscRef.current = null;
    gainRef.current = null;
  }
  function openCompanionPinger() {
    if (companionRef.current && !companionRef.current.closed) return;
    const w = window.open("/miniWindow.html", "arfocus_companion", "width=300,height=240,noopener,noreferrer");
    if (!w) return;
    companionRef.current = w;
  }
  
  
  function closeCompanionPinger() {
    try { companionRef.current?.close(); } catch {}
    companionRef.current = null;
  }
  
  // ---- Start/Pause/Reset/Save
  async function startTimer() {
    if (!isRunning) {
      const a = audioRef.current;
      if (a && a.paused) {
        try {
          a.muted = true; await a.play(); a.pause(); a.currentTime = 0; a.muted = false;
        } catch {}
      }
      setIsRunning(true);
      if (allowHiddenWork) openCompanionPinger();
      if (!sessionStartRef.current) {
        sessionStartRef.current = new Date().toISOString();
        hpStartRef.current = hp;
      }
    }
  }
  function pauseTimer() {
    setIsRunning(false);
    closeCompanionPinger();
  }
  function resetTimer() {
    setIsRunning(false);
    setIsOnBreak(false);
    setSecondsLeft(workMinutes * 60);
    setFocusSec(0);
    setDistractSec(0);
    setHp(100);
    sessionStartRef.current = null;
    const a = audioRef.current;
    if (a && !a.paused) { a.pause(); a.currentTime = 0; }
    closeCompanionPinger();
  }
  async function stopAndSave() {
    setIsRunning(false);
  
    const startedAt = sessionStartRef.current ?? new Date().toISOString();
    const endedAt   = new Date().toISOString();
  
    const duration = Math.max(1, focusSec + distractSec); // 👈 avoid 0
  
    const log: SessionLog = {
      id: `${Date.now()}`,
      startedAt,
      endedAt,
      durationSec: duration,
      workMinutes,
      breakMinutes,
      focusSec,
      distractSec,
      hpStart: hpStartRef.current,
      hpEnd: hp,
    };
  
    // local history
    const next = [log, ...sessions].slice(0, 300);
    setSessions(next);
    saveSessions(next);
  
    // cloud (either await or fire-and-forget)
    try {
      await postSessionToCloud(log);          // ← await so Leaderboard sees it right away
    } catch (err) {
      console.warn("[cloud save] failed:", err);
    }
  
    // reset for next run
    setIsOnBreak(false);
    setSecondsLeft(workMinutes * 60);
    setFocusSec(0);
    setDistractSec(0);
    setHp(100);
    sessionStartRef.current = null;
  
    const a = audioRef.current;
    if (a && !a.paused) { a.pause(); a.currentTime = 0; }
    closeCompanionPinger();
  }
  
  

  // ---- Chart data
  const chartData = useMemo(() => {
    const byDay: Record<string, { date: string; focusMin: number; distractMin: number; totalMin: number; focusRate: number }> = {};
    for (const s of sessions) {
      const key = s.startedAt.slice(0, 10);
      if (!byDay[key]) byDay[key] = { date: key, focusMin: 0, distractMin: 0, totalMin: 0, focusRate: 0 };
      byDay[key].focusMin += s.focusSec / 60;
      byDay[key].distractMin += s.distractSec / 60;
    }
    const keys = Object.keys(byDay).sort().slice(-7);
    return keys.map((k) => {
      const r = byDay[k];
      r.totalMin = Math.max(0.0001, r.focusMin + r.distractMin);
      r.focusRate = r.focusMin / r.totalMin;
      return r;
    });
  }, [sessions]);

  // ---- Camera + models + foreground/background loops
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;

    async function boot() {
      try {
        // 1) Ensure models are created once (reused across camera toggles)
        const fs = await FilesetResolver.forVisionTasks(MP_WASM_URL);

        if (!faceVideoRef.current || !faceImageRef.current) {
          faceVideoRef.current = await FaceLandmarker.createFromOptions(fs, {
            baseOptions: { modelAssetPath: MP_FACE_TASK, delegate: "GPU" },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true,
          });
          faceImageRef.current = await FaceLandmarker.createFromOptions(fs, {
            baseOptions: { modelAssetPath: MP_FACE_TASK, delegate: "GPU" },
            runningMode: "IMAGE",
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true,
          });
          setModelReady(true);
        } else {
          setModelReady(true);
        }

        if (!objVideoRef.current || !objImageRef.current) {
          async function initObj(mode: "VIDEO" | "IMAGE") {
            try {
              return await ObjectDetector.createFromOptions(fs, {
                baseOptions: { modelAssetPath: MP_OBJ_TASK, delegate: "GPU" },
                runningMode: mode,
                maxResults: 5,
                scoreThreshold: 0.25,
                categoryAllowlist: ["cell phone"],
              });
            } catch {
              return await ObjectDetector.createFromOptions(fs, {
                baseOptions: { modelAssetPath: MP_OBJ_TASK_FALLBACK, delegate: "GPU" },
                runningMode: mode,
                maxResults: 5,
                scoreThreshold: 0.25,
                categoryAllowlist: ["cell phone"],
              });
            }
          }
          objVideoRef.current = await initObj("VIDEO");
          objImageRef.current = await initObj("IMAGE");
          setObjectModelReady(true);
        } else {
          setObjectModelReady(true);
        }

        // 2) If camera is OFF, don't open it; just stop here.
        if (!cameraOn) return;

        // 3) Open camera stream
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });

        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play().catch(() => {});

        // Set real aspect ratio so the box sizes correctly (no black bands)
        const vw = v.videoWidth || 640;
        const vh = v.videoHeight || 480;
        if (vw && vh) setCamAR(vw / vh);

        // ImageCapture for background frames
        const track = stream.getVideoTracks()[0];
        if (track && window.ImageCapture) {
          try {
            imageCaptureRef.current = new window.ImageCapture(track);
          } catch {
            imageCaptureRef.current = null;
          }
        }

        if (!cancelled) startLoops();
      } catch (e: any) {
        setLastError(String(e?.message || e));
      }
    }

    function startLoops() {
      // Foreground rAF loop (only when tab is visible)
      const tick = (ts: number) => {
        if (document.hidden) {
          rafRef.current = requestAnimationFrame(tick);
          return; // background handled by interval
        }

        const v = videoRef.current!;
        const c = canvasRef.current!;
        const ctx = c.getContext("2d")!;

        // Keep canvas at exact video pixel size (overlay stays crisp)
        const W = v.videoWidth || 640;
        const H = v.videoHeight || 480;
        if (c.width !== W) c.width = W;
        if (c.height !== H) c.height = H;

        ctx.drawImage(v, 0, 0, c.width, c.height);
        stepVision({ mode: "video", t: ts, frame: v, ctx, w: c.width, h: c.height }).catch(() => {});
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);

      // Background interval loop (tab hidden)
      const bgStep = async () => {
        if (!document.hidden) return;
        const ic = imageCaptureRef.current;
        if (!ic) return;

        const c = canvasRef.current!;
        const ctx = c.getContext("2d")!;
        try {
          const bmp = await ic.grabFrame();
          if (c.width !== bmp.width) c.width = bmp.width;
          if (c.height !== bmp.height) c.height = bmp.height;
          ctx.drawImage(bmp, 0, 0, c.width, c.height);
          await stepVision({ mode: "image", t: performance.now(), frame: bmp, ctx, w: c.width, h: c.height });
          (bmp as any).close?.();
        } catch {
          // ignore frame grab errors
        }
      };
      bgIntervalRef.current = window.setInterval(bgStep, BG_INTERVAL_MS) as unknown as number;
    }

    // Boot (or re-boot) whenever the camera toggle changes
    boot();

    // Cleanup on toggle or unmount
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (bgIntervalRef.current) window.clearInterval(bgIntervalRef.current);

      const v = videoRef.current;
      if (v && v.srcObject) {
        (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        v.srcObject = null;
      }
      stream = null;
    };
  }, [cameraOn]);

  // ---- Vision step (VIDEO or IMAGE) -> updates signals and classification
  const emaRef = useRef(0);
  const frameCountRef = useRef(0);
  const timersRef = useRef({
    last: now(),
    eyesAway: 0,
    eyesNotVisible: 0,
    lookDown: 0,
    noFace: 0,
    tabInactive: 0,
    camDark: 0,
  });

  async function stepVision(opts: {
    mode: "video" | "image";
    t: number;
    frame: HTMLVideoElement | ImageBitmap;
    ctx: CanvasRenderingContext2D;
    w: number;
    h: number;
  }) {
    const { mode, t, frame, ctx, w, h } = opts;
    // Pull difficulty-dependent thresholds from cfg
    const {
      CENTER_TOL, GAZE_TOL, ENTER_TH, EXIT_TH,
      PHONE_TH, PHONE_TH_BOTTOM,
      DOWN_MILD_MS, DOWN_STRONG_MS
    } = cfgRef.current;

    // Face
    const faceV = faceVideoRef.current;
    const faceI = faceImageRef.current;
    const faceResult =
      mode === "video"
        ? faceV?.detectForVideo(frame as HTMLVideoElement, t)
        : faceI?.detect(frame as any);

    let haveFace = false;
    let localEyesVisible = false;
    let localGazeOn = false;
    let localYaw = 0, localPitch = 0;

    if (faceResult?.faceLandmarks?.length) {
      haveFace = true;
      setFacesCount(faceResult.faceLandmarks.length);

      if (faceResult.facialTransformationMatrixes?.length) {
        const matData = faceResult.facialTransformationMatrixes?.[0]?.data;
        if (matData) {
          const eul = eulerFromMat4Deg(matData);
          localYaw = eul.yawDeg;
          localPitch = eul.pitchDeg;
        }
      }

      const lm = faceResult.faceLandmarks[0];

      // bbox / center
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of lm) {
        const x = p.x * w, y = p.y * h;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
      const cx = (minX + maxX) / 2;
      const centerDx = Math.abs(cx - w / 2) / w;
      const centerOK = centerDx < CENTER_TOL;

      // FaceMesh indices
      const IDX = { R_OUT: 33, R_IN: 133, L_OUT: 263, L_IN: 362, R_IRIS: 468, L_IRIS: 473 } as const;
      const rOuter = lm[IDX.R_OUT], rInner = lm[IDX.R_IN], rI = lm[IDX.R_IRIS];
      const lOuter = lm[IDX.L_OUT], lInner = lm[IDX.L_IN], lI = lm[IDX.L_IRIS];
      localEyesVisible = !!(rOuter && rInner && rI && lOuter && lInner && lI);

      // Gaze via iris offsets
      if (localEyesVisible) {
        const normEyeOff = (lx: number, rx: number, cx: number) => {
          const width = Math.max(1, Math.abs(rx - lx));
          return Math.abs(cx - (lx + rx) / 2) / width;
        };
        const rLx = rOuter.x * w, rRx = rInner.x * w, rCx = rI.x * w;
        const lLx = lInner.x * w, lRx = lOuter.x * w, lCx = lI.x * w;
        const rOff = normEyeOff(rLx, rRx, rCx);
        const lOff = normEyeOff(lLx, lRx, lCx);
        const gazeOK = (rOff + lOff) / 2 < GAZE_TOL;

        const score = 0.2 + 0.4 * (gazeOK ? 1 : 0) + 0.4 * (centerOK ? 1 : 0);
        const ema = (emaRef.current = EMA_ALPHA * score + (1 - EMA_ALPHA) * emaRef.current);
        const prev = gazeOn;
        localGazeOn = prev === true ? ema > EXIT_TH : prev === false ? ema > ENTER_TH : ema > 0.6;

        // draw bbox
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 2;
        ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
      } else {
        localGazeOn = false;
      }

      // Eyelid openness → closed/covered eyes within 1s
      const L_TOP = 159, L_BOT = 145, R_TOP = 386, R_BOT = 374;
      const lTop = lm[L_TOP], lBot = lm[L_BOT], rTop = lm[R_TOP], rBot = lm[R_BOT];
      const dist = (a: any, b: any) => Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
      let eyesClosedThisFrame = false;
      if (lTop && lBot && rTop && rBot && rOuter && rInner && lOuter && lInner) {
        const lVert = dist(lTop, lBot), rVert = dist(rTop, rBot);
        const lHoriz = dist(lOuter, lInner), rHoriz = dist(rOuter, rInner);
        const lOpen = lVert / Math.max(1, lHoriz);
        const rOpen = rVert / Math.max(1, rHoriz);
        eyesClosedThisFrame = (lOpen < EYE_OPEN_THRESH && rOpen < EYE_OPEN_THRESH);
      }
      if (eyesClosedThisFrame) {
        eyesClosedMsRef.current += 1000 / INFER_HZ;
      } else {
        eyesClosedMsRef.current = 0;
      }
    } else {
      setFacesCount(0);
      eyesClosedMsRef.current = Math.min(EYE_CLOSED_MAX_MS, eyesClosedMsRef.current + 1000 / INFER_HZ); // treat missing eyes as closing
      localGazeOn = false;
    }

    // Object detection (phone) — run every N frames to save compute
    frameCountRef.current++;
    let sawPhoneThisFrame = false;
    const objV = objVideoRef.current, objI = objImageRef.current;
    if ((frameCountRef.current % PHONE_DETECT_EVERY) === 0) {
      const det = mode === "video"
        ? objV?.detectForVideo(frame as HTMLVideoElement, t)
        : objI?.detect(frame as any);

      if (det?.detections?.length) {
        for (const d of det.detections) {
          const cat = d.categories?.[0];
          const name = (cat?.categoryName || "").toLowerCase();
          const score = cat?.score ?? 0;
          if (name.includes("cell phone") || name === "phone") {
            const bb = d.boundingBox;
            const bottomBias = bb ? (bb.originY + bb.height) > 0.72 * h : false;
            const th = bottomBias ? Math.min(PHONE_TH, PHONE_TH_BOTTOM) : PHONE_TH;
            if (score >= th) {
              sawPhoneThisFrame = true;
              lastPhoneSeenAtRef.current = performance.now();
              if (bb) {
                ctx.strokeStyle = "#ef4444";
                ctx.lineWidth = 2;
                ctx.strokeRect(bb.originX, bb.originY, bb.width, bb.height);
                ctx.fillStyle = "rgba(239,68,68,0.16)";
                ctx.fillRect(bb.originX, bb.originY, bb.width, bb.height);
              }
              break;
            }
          }
        }
      }
    }
    const phonePersist = performance.now() - lastPhoneSeenAtRef.current <= 1000;
    const phoneDetected = sawPhoneThisFrame || phonePersist;
    setPhoneNow(phoneDetected);

    // Camera covered heuristic
    const dark = isVeryDark(ctx, w, h);
    setCamCovered(dark);

    // Publish instantaneous signals
    setGazeOn(localGazeOn);
    setEyesVisible(!!localEyesVisible);
    setHeadYaw(localYaw);
    setHeadPitch(localPitch);
    setLookingDown(localPitch > 15);

    // Timers & classification
    const nowT = now();
    const dt = nowT - timersRef.current.last;
    timersRef.current.last = nowT;

    if (!haveFace) timersRef.current.noFace = Math.min(NO_FACE_STRONG_MS + 2000, timersRef.current.noFace + dt);
    else timersRef.current.noFace = 0;

    if (!eyesVisible) timersRef.current.eyesNotVisible = Math.min(EYES_NOT_VISIBLE_MS + 2000, timersRef.current.eyesNotVisible + dt);
    else timersRef.current.eyesNotVisible = 0;

    const headAbsYaw = Math.abs(localYaw);
    const headAbsPitch = Math.abs(localPitch);
    const headWithin15 = headAbsYaw <= 15 && headAbsPitch <= 15;
    const head15to45 = (!headWithin15) && (headAbsYaw <= 45 && headAbsPitch <= 45);
    const headOver45 = headAbsYaw > 45 || headAbsPitch > 45;

    const eyesAway = !localGazeOn;
    if (eyesAway) timersRef.current.eyesAway = Math.min(EYES_AWAY_MIN_MS + 2000, timersRef.current.eyesAway + dt);
    else timersRef.current.eyesAway = 0;

    if (localPitch > 15) timersRef.current.lookDown = Math.min(DOWN_STRONG_MS + 2000, timersRef.current.lookDown + dt);
    else timersRef.current.lookDown = 0;

    // Tab hidden — allow multi-tab work if (a) we still capture frames or (b) companion/pings are fresh
    const nowMs = performance.now();
    const freshProductive = (nowMs - lastProductivePingRef.current) <= PRODUCTIVE_PING_FRESH_MS;

    // If camera still delivers frames while hidden, treat as OK
    const freshCapture = (nowMs - lastVisionAtRef.current) <= Math.max(1000, BG_INTERVAL_MS * 3);

    // Hidden is OK when multi-tab allowed AND we have either a ping or fresh capture
    const hiddenOk = allowHiddenWork && (freshProductive || freshCapture);

    if (document.hidden && !hiddenOk) {
      timersRef.current.tabInactive = Math.min(8000, timersRef.current.tabInactive + dt);
    } else {
      timersRef.current.tabInactive = 0;
    }

    if (dark) timersRef.current.camDark = Math.min(NO_FACE_STRONG_MS + 2000, timersRef.current.camDark + dt);
    else timersRef.current.camDark = 0;

    // Decide level (max severity)
    let nextLevel: DistractionLevel = "Focused";

    // Critical
    if (
      phoneDetected ||
      timersRef.current.tabInactive > 3000 ||
      timersRef.current.noFace > 3000 ||
      timersRef.current.camDark > 2000
    ) {
      nextLevel = "Critical";
    } else {
      // Strong
      if (
        eyesClosedMsRef.current >= EYE_CLOSED_MAX_MS ||
        timersRef.current.eyesNotVisible > EYES_NOT_VISIBLE_MS ||
        headOver45 ||
        timersRef.current.lookDown > DOWN_STRONG_MS ||
        phoneDetected ||
        timersRef.current.tabInactive > 2000
      ) {
        nextLevel = "Strong";
      }
      // Mild
      else if (
        timersRef.current.eyesAway > EYES_AWAY_MIN_MS ||
        head15to45 ||
        timersRef.current.lookDown > DOWN_MILD_MS ||
        timersRef.current.tabInactive > 1000
      ) {
        nextLevel = "Mild";
      } else {
        // Strict focus
        if (!(localGazeOn && headWithin15 && !phoneDetected && !dark && haveFace)) {
          nextLevel = "Focused";
        }
      }
    }

    lastVisionAtRef.current = performance.now();
    setLevel(nextLevel);
  }

  // ---- Progress ring
  const totalPhaseSec = (isOnBreak ? breakMinutes : workMinutes) * 60;
  const pct = ((totalPhaseSec - secondsLeft) / totalPhaseSec) * 100;

  const levelColor =
    level === "Focused" ? "text-emerald-600" :
    level === "Mild"    ? "text-amber-600"  :
    level === "Strong"  ? "text-rose-600"   :
                          "text-red-700";

  // ---- Theme helpers (dark/light) ----
  const isDark = (profile?.theme === "dark");

  useEffect(() => {
    // make native controls/scrollbars follow theme
    document.documentElement.style.colorScheme = isDark ? "dark" : "light";
  }, [isDark]);

  // Reusable class groups
  const Pane   = `rounded-2xl shadow ${isDark ? "bg-slate-900/70 text-slate-100" : "bg-white text-slate-900"} p-4`;
  const Soft   = `${isDark ? "bg-slate-800/60" : "bg-slate-100"}`;
  const Subtle = `${isDark ? "text-slate-400" : "text-slate-500"}`;
  const Accent = `${isDark ? "text-indigo-300" : "text-indigo-600"}`;
  // When camera turns off, blank the canvas to pure black
  useEffect(() => {
    if (cameraOn) return;
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    // ensure the canvas has a size
    if (!c.width) c.width = 640;
    if (!c.height) c.height = 480;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
  }, [cameraOn]);

  // ---- UI
  return (
    <div className={`min-h-screen w-full ${isDark ? "bg-slate-950 text-slate-100" : "bg-slate-50 text-slate-900"}`}>
      <audio ref={audioRef} src="/distraction.mp3" preload="auto" />
  
      <div className="max-w-5xl mx-auto px-4 py-6">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-2xl bg-indigo-100 text-indigo-700">
              <Camera size={18} />
            </div>
            <h1 className="text-xl font-semibold">AR Focus App — Full MVP</h1>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1">
              <ShieldCheck size={14} />
              <span>On-device only • No recording</span>
            </div>
          </div>
        </header>
  
        <DailyMotivationBanner quote={motivation} />
  
        {/* top pills */}
        <div className="flex gap-2 justify-center mt-2">
          <button
            onClick={() => setShowPrivacy(true)}
            className={`px-3 py-1 rounded-xl text-sm ${Soft}`}
          >
            Privacy
          </button>
          <button
            onClick={() => setShowCustomize(true)}
            className={`px-3 py-1 rounded-xl text-sm ${Soft}`}
          >
            Customize
          </button>
          <button
            onClick={() => setShowLeaderboard(true)}
            className={`px-3 py-1 rounded-xl text-sm ${Soft}`}
          >
            Leaderboard
          </button>
        </div>

        {/* modals – pass isDark */}
        {showPrivacy && (
          <PrivacyDashboard isDark={isDark} onClose={() => setShowPrivacy(false)} />
        )}
        {showCustomize && (
          <CustomizationPanel
            isDark={isDark}
            profile={profile}
            updateProfile={updateProfile}
            onClose={() => setShowCustomize(false)}
          />
        )}
        {showLeaderboard && <Leaderboard isDark={isDark} onClose={() => setShowLeaderboard(false)} />}

        {/* Controls */}
        <section className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Focus ring card */}
          <div className={`col-span-1 ${Pane} flex flex-col items-center`}>
            <div className="relative w-44 h-44">
              <svg className="w-44 h-44 -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" stroke="#e5e7eb" strokeWidth="10" fill="none" />
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  stroke={isOnBreak ? "#f59e0b" : "#6366f1"}
                  strokeWidth="10"
                  fill="none"
                  strokeDasharray={`${(pct/100)*2*Math.PI*45} ${2*Math.PI*45}`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-3xl font-bold tabular-nums">{formatMMSS(secondsLeft)}</div>
                <div className={`text-xs mt-1 ${isOnBreak ? "text-amber-600" : "text-indigo-600"}`}>
                  {isOnBreak ? "Break" : "Focus"}
                </div>
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
  
            <div className={`mt-3 text-xs ${Subtle} flex items-center gap-2`}>
              <Info size={14} /> Focus adds HP slowly; distractions reduce it faster.
            </div>
          </div>
  
          {/* Settings */}
          <div className={`col-span-1 ${Pane}`}>
            <h2 className="font-semibold mb-3">Session Settings</h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm">Work (min)</label>
                <input
                  type="number"
                  min={5}
                  max={120}
                  value={workMinutes}
                  onChange={(e) => setWorkMinutes(Math.max(5, Math.min(120, Number(e.target.value))))}
                  className={`w-24 px-2 py-1 rounded-xl ${Soft}`}
                />
              </div>
              <div className="flex items-center justify-between">
                <label className="text-sm">Break (min)</label>
                <input
                  type="number"
                  min={3}
                  max={60}
                  value={breakMinutes}
                  onChange={(e) => setBreakMinutes(Math.max(3, Math.min(60, Number(e.target.value))))}
                  className={`w-24 px-2 py-1 rounded-xl ${Soft}`}
                />
              </div>
  
              {/* Beep toggle + volume */}
              <div className="flex items-center justify-between">
                <label className="text-sm flex items-center gap-2">
                  {beepOnDistract ? <Volume2 size={14} /> : <VolumeX size={14} />}
                  Beep on distraction
                </label>
                <button
                  onClick={() => setBeepOnDistract((v) => !v)}
                  className={`px-3 py-1 rounded-xl ${
                    beepOnDistract ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-900"}`}>
                  {beepOnDistract ? "On" : "Off"}
                </button>

              </div>
              <div className="flex items-center justify-between">
                <label className="text-sm">Beep volume</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={beepVolume}
                  onChange={(e) => setBeepVolume(Number(e.target.value))}
                  className="w-32"
                />
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm">Camera</label>
                <button
                  onClick={() => setCameraOn((v) => !v)}
                  className={`px-3 py-1 rounded-xl ${
                    cameraOn ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-900"}`}>
                  {cameraOn ? "On" : "Off"}
                </button>
              </div>

              <div className="flex items-center justify-between">
              <label className="text-sm">Mirror camera</label>
                <button
                  onClick={() => setMirrorCam((v) => !v)}
                  className={`px-3 py-1 rounded-xl ${
                    mirrorCam
                      ? "bg-emerald-600 text-white"
                      : isDark
                        ? "bg-slate-800 text-slate-100"
                        : "bg-slate-200 text-slate-900"}`}>
                  {mirrorCam ? "On" : "Off"}
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm">Multi-tab mode (don’t penalize hidden tabs)</label>
                <button
                  onClick={() => setAllowHiddenWork(v => !v)}
                  className={`px-3 py-1 rounded-xl ${
                    allowHiddenWork ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-900"
                  }`}
                >
                  {allowHiddenWork ? "On" : "Off"}
                </button>
              </div>


              <div className={`text-xs ${Subtle} leading-relaxed`}>
                • All processing is local in your browser. • No video is recorded or uploaded.
              </div>
            </div>
          </div>
  
          {/* Live status */}
          <div className={`col-span-1 ${Pane}`}>
            <h2 className="font-semibold mb-3">Live Status</h2>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {/* HP */}
              <div className={`p-3 rounded-xl ${Soft} flex flex-col`}>
                <span className={Subtle}>HP</span>
                <span className="font-semibold">{hp.toFixed(1)}</span>
              </div>
              {/* Attention */}
              <div className={`p-3 rounded-xl ${Soft} flex flex-col`}>
                <span className={Subtle}>Attention</span>
                <span className={`text-sm font-semibold ${levelColor}`}>{level}</span>
              </div>
              {/* Focused / Distracted seconds */}
              <div className={`p-3 rounded-xl ${Soft} flex flex-col`}>
                <span className={Subtle}>Focused (s)</span>
                <span className="font-semibold">{focusSec}</span>
              </div>
              <div className={`p-3 rounded-xl ${Soft} flex flex-col`}>
                <span className={Subtle}>Distracted (s)</span>
                <span className="font-semibold">{distractSec}</span>
              </div>
  
              {/* Phone tile */}
              <div className={`p-3 rounded-xl ${Soft} flex flex-row items-center gap-2`}>
                <Smartphone size={16} className={phoneNow ? "text-rose-600" : "text-slate-400"} />
                <div className="flex flex-col">
                  <span className={Subtle}>Phone</span>
                  <span className={`text-xs font-semibold ${phoneNow ? "text-rose-500" : Subtle}`}>
                    {objectModelReady ? (phoneNow ? "Detected" : "None") : "Model unavailable"}
                  </span>
                </div>
              </div>
            </div>
            <div className={`mt-3 text-xs ${Subtle}`}>
              {modelReady ? "Face model active" : "Face model not loaded"} • {objectModelReady ? "Phone model active" : "Phone model not loaded"}
            </div>
            {lastError && <div className="mt-2 text-[11px] text-rose-600">Err: {lastError}</div>}
          </div>
        </section>
  
        {/* Camera feed + chart */}
        <section className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Camera box — correct aspect, no stretching, no black band */}
        <div className={`${isDark ? "bg-black/80" : "bg-black"} rounded-2xl shadow`}>
          {/* The inner wrapper defines the height using CSS aspect-ratio */}
          <div className="relative w-full overflow-hidden rounded-2xl"
              style={{ aspectRatio: camAR }}>
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover opacity-70 ${cameraOn ? "" : "hidden"} ${
                mirrorCam ? "-scale-x-100" : ""
              }`}
            />

            <canvas
              ref={canvasRef}
              className={`absolute inset-0 w-full h-full pointer-events-none ${
                mirrorCam ? "-scale-x-100" : ""
              }`}
            />

            {(!modelReady || !objectModelReady) && (
              <div className="absolute inset-0 flex items-center justify-center text-white/80 text-sm">
                Initializing models…
              </div>
            )}
          </div>
        </div>

        {/* Chart */}
        <div className={`${Pane}`}>
          <h2 className={`font-semibold mb-3 flex items-center gap-2 ${Accent}`}>
            <TrendingUp size={18} /> Weekly Focus vs Distraction
          </h2>
          <div className="w-full" style={{ height: 260, minWidth: 320 }} key={chartData.map(d => d.date).join("|")}>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="min" tick={{ fontSize: 12 }} label={{ value: "Minutes", angle: -90, position: "insideLeft" }} allowDecimals={false} />
                  <YAxis yAxisId="rate" orientation="right" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip formatter={(value: any, name: any) => {
                    if (name === "Focus rate") return [`${Math.round((value as number) * 100)}%`, name];
                    return [Math.round(value as number), name];
                  }} />
                  <Legend />
                  <Bar yAxisId="min" dataKey="distractMin" name="Distract (min)" stackId="a" fill="#ef4444" opacity={0.65} />
                  <Bar yAxisId="min" dataKey="focusMin"   name="Focus (min)"   stackId="a" fill="#22c55e" opacity={0.75} />
                  <Line yAxisId="rate" type="monotone" dataKey="focusRate" name="Focus rate" dot={false} stroke="#3b82f6" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className={`h-full flex items-center justify-center text-sm ${Subtle}`}>
                No data yet — press <span className="mx-1 font-medium">Start</span>, then <span className="mx-1 font-medium">Save</span>.
              </div>
            )}
          </div>
          <p className={`mt-2 text-xs ${Subtle}`}>Bars: minutes per day (stacked). Line: % time focused.</p>
        </div>
        </section>

  
        {/* History & Export */}
        <section className={`mt-6 ${Pane}`}>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Saved Sessions</h2>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  const header = ["Date","Start","End","Duration","Focus","Distract","Focus %","HP Δ"];
                  const rows = sessions.map((s) => {
                    const start = new Date(s.startedAt);
                    const end = new Date(s.endedAt);
                    const focusPct = (s.focusSec + s.distractSec) > 0 ? s.focusSec / (s.focusSec + s.distractSec) : 0;
                    const hpDelta = s.hpEnd - s.hpStart;
                    return [
                      start.toISOString().slice(0,10),
                      start.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                      end.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                      mmss(s.durationSec),
                      mmss(s.focusSec),
                      mmss(s.distractSec),
                      `${Math.round(focusPct*100)}%`,
                      hpDelta.toFixed(1),
                    ].join(",");
                  });
                  const csv = [header.join(","), ...rows].join("\n");
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
  
              <button
                onClick={() => {
                  const csv = toCSV(sessions);
                  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `arfocus_sessions_raw_${Date.now()}.csv`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                className="px-3 py-2 rounded-xl bg-slate-200 text-slate-900"
              >
                Export Raw
              </button>
            </div>
          </div>
  
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className={Subtle}>
                <tr className="text-left">
                  <th className="py-2 pr-4">Date</th>
                  <th className="py-2 pr-4">Start</th>
                  <th className="py-2 pr-4">End</th>
                  <th className="py-2 pr-4">Dur</th>
                  <th className="py-2 pr-4">Focus</th>
                  <th className="py-2 pr-4">Distract</th>
                  <th className="py-2 pr-4">Focus %</th>
                  <th className="py-2 pr-4">HP Δ</th>
                </tr>
              </thead>
              <tbody>
                {sessions.length === 0 && (
                  <tr>
                    <td className={`py-3 ${Subtle}`} colSpan={8}>
                      No sessions yet — press Start, then Save.
                    </td>
                  </tr>
                )}
                {sessions.map((s) => {
                  const focusPct = (s.focusSec + s.distractSec) > 0 ? s.focusSec / (s.focusSec + s.distractSec) : 0;
                  const hpDelta = s.hpEnd - s.hpStart;
                  const start = new Date(s.startedAt);
                  const end = new Date(s.endedAt);
                  return (
                    <tr key={s.id} className={isDark ? "border-t border-slate-800/60" : "border-t border-slate-100"}>
                      <td className="py-2 pr-4">{formatLocal(s.startedAt).split(",")[0]}</td>
                      <td className="py-2 pr-4">{start.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
                      <td className="py-2 pr-4">{end.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
                      <td className="py-2 pr-4">{mmss(s.durationSec)}</td>
                      <td className="py-2 pr-4">{mmss(s.focusSec)}</td>
                      <td className="py-2 pr-4">{mmss(s.distractSec)}</td>
                      <td className="py-2 pr-4">{fmtPct(focusPct)}</td>
                      <td className={`py-2 pr-4 font-medium ${hpDelta >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                        {hpDelta >= 0 ? `+${hpDelta.toFixed(1)}` : hpDelta.toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              {sessions.length > 0 && (() => {
                const totals = sessions.reduce(
                  (acc, s) => {
                    acc.dur += s.durationSec;
                    acc.foc += s.focusSec;
                    acc.dis += s.distractSec;
                    acc.hp += s.hpEnd - s.hpStart;
                    return acc;
                  },
                  { dur: 0, foc: 0, dis: 0, hp: 0 }
                );
                const focusPct = (totals.foc + totals.dis) > 0 ? totals.foc / (totals.foc + totals.dis) : 0;
                return (
                  <tfoot>
                    <tr className={isDark ? "border-t border-slate-800/60 font-medium" : "border-t border-slate-200 font-medium"}>
                      <td className="py-2 pr-4" colSpan={3}>Totals</td>
                      <td className="py-2 pr-4">{mmss(totals.dur)}</td>
                      <td className="py-2 pr-4">{mmss(totals.foc)}</td>
                      <td className="py-2 pr-4">{mmss(totals.dis)}</td>
                      <td className="py-2 pr-4">{fmtPct(focusPct)}</td>
                      <td className={`py-2 pr-4 ${totals.hp >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                        {totals.hp >= 0 ? `+${totals.hp.toFixed(1)}` : totals.hp.toFixed(1)}
                      </td>
                    </tr>
                  </tfoot>
                );
              })()}
            </table>
          </div>
        </section>
      </div>
    </div>
  );  
}

// ----------------------------- Small UI helper -----------------------------
function StatCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="p-3 rounded-xl bg-slate-100 flex flex-col">
      <span className="text-slate-500">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
// ==========================================================================
// 🧩 PART 2 — Privacy Dashboard • Customization • Leaderboard • Motivation
// ==========================================================================

// ---------- Privacy Dashboard ----------
export function PrivacyDashboard({
  onClose,
  isDark,
}: {
  onClose: () => void;
  isDark: boolean;
}) {
  const pane   = isDark ? "bg-slate-900/90 text-slate-100" : "bg-white text-slate-900";
  const subtle = isDark ? "text-slate-400"                 : "text-slate-600";

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className={`${pane} rounded-2xl shadow-xl max-w-lg w-full p-6 relative`}>
        <button
          className={`absolute top-3 right-3 ${subtle} hover:!text-slate-700`}
          onClick={onClose}
        >
          <X size={18} />
        </button>
        <h2 className="text-lg font-semibold mb-2">Privacy Dashboard</h2>
        <p className={`text-sm ${subtle} leading-relaxed`}>
          • The camera feed is processed <strong>entirely on-device</strong> using MediaPipe and
          never leaves your browser. <br />
          • No images, videos, or biometric data are uploaded or stored. <br />
          • Only session statistics (focus time, HP, etc.) are optionally synced to Supabase if you
          enable cloud sync. <br />
          • You can clear all local data anytime by pressing <code>localStorage.clear()</code> in your console.
        </p>
      </div>
    </div>
  );
}

// ---------- Customization Panel ----------
export function CustomizationPanel({
  profile,
  updateProfile,
  onClose,
  isDark,
}: {
  profile: any;
  updateProfile: (p: any) => void;
  onClose: () => void;
  isDark: boolean;
}) {
  if (!profile) return null;

  const pane   = isDark ? "bg-slate-900/90 text-slate-100" : "bg-white text-slate-900";
  const soft   = isDark ? "bg-slate-800/60 text-slate-100" : "bg-slate-100 text-slate-900";
  const subtle = isDark ? "text-slate-400"                 : "text-slate-600";

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className={`${pane} rounded-2xl shadow-xl max-w-md w-full p-6 relative`}>
        <button
          className={`absolute top-3 right-3 ${subtle} hover:!text-slate-700`}
          onClick={onClose}
        >
          <X size={18} />
        </button>

        <h2 className="text-lg font-semibold mb-4">Customization</h2>
        <div className="space-y-4 text-sm">
          <div>
            <label className={`block mb-1 ${subtle}`}>Theme</label>
            <select
              value={profile.theme}
              onChange={(e) => updateProfile({ theme: e.target.value })}
              className={`w-full px-3 py-2 rounded-xl ${soft}`}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>

          <div>
            <label className={`block mb-1 ${subtle}`}>Difficulty</label>
            <select
              value={profile.difficulty}
              onChange={(e) => updateProfile({ difficulty: e.target.value })}
              className={`w-full px-3 py-2 rounded-xl ${soft}`}
            >
              <option value="easy">Easy</option>
              <option value="normal">Normal</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <div className={`pt-2 text-xs ${subtle}`}>
            Difficulty now tunes model strictness and default work/break lengths; theme changes visuals only.
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- Daily Motivation Banner ----------
export function DailyMotivationBanner({ quote }: { quote: string }) {
  if (!quote) return null;
  return (
    <div className="mt-4 text-center">
      <div className="inline-block bg-indigo-50 text-indigo-700 px-4 py-2 rounded-full text-sm font-medium shadow-sm">
        💡 {quote}
      </div>
    </div>
  );
}

