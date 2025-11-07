import React from "react";
import { pickTip } from "../lib/tips";
export default function TipsWidget() {
  return (
    <div className="p-3 rounded-xl bg-indigo-50 text-indigo-800 text-xs">
      💡 {pickTip()}
    </div>
  );
}
