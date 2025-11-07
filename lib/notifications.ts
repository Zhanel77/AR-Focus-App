export async function ensurePermission(): Promise<boolean> {
    if (!("Notification" in window)) return false;
    if (Notification.permission === "granted") return true;
    if (Notification.permission !== "denied") {
      const p = await Notification.requestPermission();
      return p === "granted";
    }
    return false;
  }
  export function notify(title: string, body: string) {
    try { if ("Notification" in window && Notification.permission === "granted") new Notification(title, { body }); } catch {}
  }
  