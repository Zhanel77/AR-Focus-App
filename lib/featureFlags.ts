// Freemium gates and “Free Exam Season” window.
// Start with a simple date-based switch; later swap to backend config.
const EXAM_MONTHS = new Set([5, 12]); // May, Dec (0-indexed month+1)
export function isExamSeason(d = new Date()) {
  // Option A: simple month rule
  const m = d.getMonth() + 1;
  return EXAM_MONTHS.has(m);
  // Option B later: fetch live windows from backend /university calendars
}

export type Plan = "free" | "premium";
export function currentPlan(): Plan {
  // later: read from auth/entitlements; for now from localStorage
  const v = localStorage.getItem("arfocus.plan");
  return v === "premium" ? "premium" : "free";
}
export function hasPremium(): boolean {
  return currentPlan() === "premium" || isExamSeason();
}
