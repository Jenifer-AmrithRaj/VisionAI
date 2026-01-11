// static/js/main.js — VisionAI Frontend Logic

// ===== Image Preview =====
function previewImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const img = document.getElementById("preview-img");
    const msg = document.getElementById("preview-msg");
    img.src = e.target.result;
    img.style.display = "block";
    msg.style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ===== Loading Overlay =====
function showLoading(msg = "Analyzing image...") {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) {
    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="text-center">
        <div class="spinner-border text-info mb-3" role="status"></div>
        <div>${msg}</div>
      </div>`;
  }
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.style.display = "none";
}

// ===== Camera Capture =====
let stream;
async function toggleCamera() {
  const video = document.getElementById("camera");
  const button = document.getElementById("open-camera");

  if (video.style.display === "none" || !video.srcObject) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
      });
      video.srcObject = stream;
      video.style.display = "block";
      button.textContent = "📸 Capture Snapshot";
    } catch (err) {
      alert("Camera error: " + err.message);
    }
  } else {
    const canvas = document.getElementById("snapshot");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const input = document.querySelector('input[name="fundus_image"]');
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      previewImage({ target: input });
    }, "image/jpeg", 0.9);

    // Stop camera
    if (stream) stream.getTracks().forEach((t) => t.stop());
    video.style.display = "none";
    button.textContent = "Open Camera";
  }
}

// ===== PWA Install Prompt =====
let deferredPrompt;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredPrompt = e;
  console.log("💾 VisionAI install prompt ready.");
  const installBtn = document.getElementById("install-btn");
  if (installBtn) installBtn.style.display = "inline-block";
});

function installPWA() {
  if (deferredPrompt) {
    deferredPrompt.prompt();
    deferredPrompt.userChoice.then((choice) => {
      console.log("📱 PWA install result:", choice.outcome);
      deferredPrompt = null;
    });
  } else {
    alert("App already installed or install prompt not ready yet!");
  }
}

// ===== Service Worker Registration =====
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/static/js/sw.js")
    .then(() => console.log("✅ VisionAI Service Worker registered!"))
    .catch((err) => console.warn("⚠️ Service Worker registration failed:", err));
}
