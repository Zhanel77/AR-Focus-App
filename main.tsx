import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// Mount the React app into the <div id="root"> in index.html
const root = createRoot(document.getElementById("root")!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
