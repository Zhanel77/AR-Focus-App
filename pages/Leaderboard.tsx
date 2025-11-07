import React, { useEffect, useState } from "react";
import { fetchWeeklyLeaderboard } from "@/lib/api";

export default function Leaderboard() {
  const [rows, setRows] = useState<any[]>([]);
  useEffect(() => { fetchWeeklyLeaderboard().then(setRows); }, []);
  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold mb-3">Weekly Leaderboard</h2>
      <table className="min-w-full text-sm">
        <thead><tr className="text-left text-slate-500">
          <th className="py-2 pr-4">User</th><th className="py-2 pr-4">Focus (min)</th><th className="py-2 pr-4">Week</th>
        </tr></thead>
        <tbody>
          {rows.map((r,i)=>(
            <tr key={i} className="border-t border-slate-100">
              <td className="py-2 pr-4">{r.user}</td>
              <td className="py-2 pr-4">{r.focusMin}</td>
              <td className="py-2 pr-4">{r.week}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
