// File: static/js/sidebar_monitor.js
const cpuEl = document.getElementById("sidebar-cpu");
const ramEl = document.getElementById("sidebar-ram");
const gpuEl = document.getElementById("sidebar-gpu");
const tempEl = document.getElementById("sidebar-temp");
const toggle = document.getElementById("sidebar-toggle");
const sidebar = document.getElementById("visionai-sidebar");

async function updateStats() {
  try {
    const res = await fetch("/api/system_stats");
    if (!res.ok) return;
    const j = await res.json();
    if (cpuEl) cpuEl.textContent = j.cpu + "%";
    if (ramEl) ramEl.textContent = j.ram + "%";
    if (gpuEl) gpuEl.textContent = j.gpu + " MB";
    if (tempEl) tempEl.textContent = j.temp + "°C";
  } catch (e) {
    // ignore
  }
}

setInterval(updateStats, 2000);
updateStats();

if (toggle && sidebar) {
  toggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });
}
