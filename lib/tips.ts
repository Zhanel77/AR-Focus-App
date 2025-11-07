export const TIPS: string[] = [
    "Two minutes of breathing can reset your focus.",
    "Phones face-down and out of reach improves attention.",
    "Short breaks protect long-term stamina.",
    "Write a tiny goal for the next 10 minutes.",
    "Full-screen the window to reduce visual noise."
  ];
  export function pickTip(seed = Date.now()): string {
    const i = Math.abs(Math.floor(seed / (1000*60*60*24))) % TIPS.length;
    return TIPS[i];
  }
  