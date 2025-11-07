import React from "react";
export default function Paywall({ feature }: { feature: string }) {
  return (
    <div className="p-4 rounded-xl bg-slate-100 text-slate-700 text-sm">
      <b>{feature}</b> is a Premium feature. Unlock Premium or wait for “Free Exam Season” to try everything for free.
    </div>
  );
}
