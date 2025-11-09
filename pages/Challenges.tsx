import React, { useEffect, useState } from "react";
import { fetchActiveChallenges } from "../lib/api";

export default function Challenges() {
  const [list, setList] = useState<any[]>([]);
  useEffect(()=>{ fetchActiveChallenges().then(setList); },[]);
  return (
    <div className="p-4 space-y-3">
      <h2 className="text-lg font-semibold">Challenges</h2>
      {list.map(c => (
        <div key={c.id} className="p-3 rounded-xl bg-slate-100 flex items-center justify-between">
          <div>
            <div className="font-medium">{c.title}</div>
            <div className="text-xs text-slate-500">Goal: {c.goalMin} min</div>
          </div>
          <button className={`px-3 py-1 rounded-xl ${c.joined ? "bg-emerald-600 text-white" : "bg-slate-300"}`}>
            {c.joined ? "Joined" : "Join"}
          </button>
        </div>
      ))}
    </div>
  );
}
