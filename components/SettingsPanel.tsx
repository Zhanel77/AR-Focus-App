import React, { useEffect, useState } from "react";
import { getSettings, subscribeSettings } from "../store/settings";

export default function SettingsPanel() {
  const [s, setS] = useState(getSettings());
  useEffect(() => subscribeSettings(() => setS(getSettings())), []); // ✅ fine, since it returns a function

  return (
    <div className="p-4 space-y-3 text-sm">
      <h2 className="text-lg font-semibold">Customization</h2>
      <div className="flex items-center justify-between">
        <label>Theme</label>
        <select
          value={s.theme}
          onChange={(e) => setS({ ...s, theme: e.target.value as any })}
          className="px-2 py-1 rounded-xl bg-slate-100"
        >
          <option value="demonic">Demonic</option>
          <option value="angelic">Angelic</option>
        </select>
      </div>
      <div className="flex items-center justify-between">
        <label>Difficulty</label>
        <select
          value={s.difficulty}
          onChange={(e) => setS({ ...s, difficulty: e.target.value as any })}
          className="px-2 py-1 rounded-xl bg-slate-100"
        >
          <option value="easy">Easy (short sessions, lenient yaw/pitch)</option>
          <option value="normal">Normal</option>
          <option value="hard">Hard (long sessions, strict yaw/pitch)</option>
        </select>
      </div>
      <div className="flex items-center justify-between">
        <label>Daily tips</label>
        <input
          type="checkbox"
          checked={s.tipsOn}
          onChange={(e) => setS({ ...s, tipsOn: e.target.checked })}
        />
      </div>
    </div>
  );
}
