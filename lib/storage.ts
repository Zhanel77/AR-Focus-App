export const read = <T>(k: string, d: T): T => {
    try { const v = localStorage.getItem(k); return v ? JSON.parse(v) as T : d; } catch { return d; }
  };
  export const write = (k: string, v: any) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };
  