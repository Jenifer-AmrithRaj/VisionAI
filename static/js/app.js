/* File: static/js/app.js
   Purpose: Core JS for VisionAI PWA — handles install prompt, animations, and service worker updates.
*/

document.addEventListener("DOMContentLoaded", () => {
  const installBtn = document.getElementById("install-btn");
  const installCard = document.getElementById("install-card");
  let deferredPrompt;

  // ✅ Hero text fade-in
  const heroLeft = document.querySelector(".hero-left");
  if (heroLeft) {
    heroLeft.style.opacity = 0;
    setTimeout(() => {
      heroLeft.style.transition = "opacity 1.2s ease-in-out";
      heroLeft.style.opacity = 1;
    }, 300);
  }

  // ✅ Handle PWA install event
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (installCard) {
      installCard.style.display = "block";
      installCard.style.opacity = 0;
      setTimeout(() => (installCard.style.opacity = 1), 200);
    }
  });

  if (installBtn) {
    installBtn.addEventListener("click", async () => {
      if (!deferredPrompt) {
        alert("App is already installable or installed.");
        return;
      }

      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;

      if (outcome === "accepted") {
        console.log("✅ VisionAI installed successfully!");
        installCard.style.display = "none";
      } else {
        console.log("Installation dismissed by user.");
      }

      deferredPrompt = null;
    });
  }

  // ✅ Service Worker Update Listener
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      console.log("🔄 New version of VisionAI available, refreshing...");
      // Auto-refresh when a new version is available
      window.location.reload();
    });
  }

  console.log("✅ VisionAI app.js loaded successfully");
});
