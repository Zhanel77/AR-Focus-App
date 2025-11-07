import React from "react";
import { hasPremium, isExamSeason } from "../lib/featureFlags";

export default function PrivacyDashboard() {
  return (
    <div className="p-4 space-y-3 text-sm">
      <h2 className="text-lg font-semibold">Privacy & Data</h2>
      <ul className="list-disc pl-5 space-y-1">
        <li>All processing runs <b>locally</b> in your browser using MediaPipe. No video is uploaded.</li>
        <li>No raw frames are stored. Only session statistics (seconds focused/distracted) are saved locally.</li>
        <li>You may export or delete your data anytime from the Sessions table.</li>
      </ul>
      <div className="text-xs text-slate-500">
        Plan: {hasPremium() ? "Premium (or Exam Season unlock)" : "Free"} •
        Exam season active: {isExamSeason() ? "Yes" : "No"}
      </div>
    </div>
  );
}
