// Minimal typing so TS stops complaining
interface ImageCapture {
  constructor(videoTrack: MediaStreamTrack): ImageCapture;
  grabFrame(): Promise<ImageBitmap>;
  takePhoto?(photoSettings?: any): Promise<Blob>;
}
interface Window { ImageCapture?: { new(track: MediaStreamTrack): ImageCapture } }
export {};
