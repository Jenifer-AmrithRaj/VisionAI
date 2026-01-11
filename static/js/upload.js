// File: static/js/upload.js
// Handles image upload, camera capture, live preview, and robust client-side validation

const imageUpload = document.getElementById("imageUpload");
const previewImage = document.getElementById("previewImage");
const previewContainer = document.getElementById("previewContainer");
const uploadForm = document.getElementById("uploadForm");

if (imageUpload) {
  imageUpload.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (event) {
        previewImage.src = event.target.result;
        previewContainer.style.display = "block";
      };
      reader.readAsDataURL(file);
    }
  });
}

// Camera capture
const captureBtn = document.getElementById("captureBtn");
const capturePhotoBtn = document.getElementById("capturePhotoBtn");
const cameraStream = document.getElementById("cameraStream");
const canvas = document.getElementById("canvas");

if (captureBtn && capturePhotoBtn && cameraStream && canvas) {
  let stream = null;
  captureBtn.addEventListener("click", async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      cameraStream.srcObject = stream;
      cameraStream.style.display = "block";
      capturePhotoBtn.style.display = "inline-block";
    } catch (err) {
      alert("Camera access denied or unavailable.");
    }
  });

  capturePhotoBtn.addEventListener("click", () => {
    const ctx = canvas.getContext("2d");
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const file = new File([blob], "captured_image.png", { type: "image/png" });
      previewImage.src = URL.createObjectURL(blob);
      previewContainer.style.display = "block";

      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      imageUpload.files = dataTransfer.files;
    });

    stream.getTracks().forEach((t) => t.stop());
    cameraStream.style.display = "none";
    capturePhotoBtn.style.display = "none";
  });
}

// Smooth validation: red border + scroll to first invalid
if (uploadForm) {
  uploadForm.addEventListener("submit", function (e) {
    // List of required fields (all 15 + image)
    const requiredFields = [
      "Full_Name","Age","Gender","Systolic","Diastolic",
      "Glucose_Level","BMI","Duration","Smoking",
      "Hypertension","Family_History","Cholesterol","HbA1c",
      "Insulin_Use","Physical_Activity","Medication"
    ];

    // remove previous error highlights
    requiredFields.forEach((n) => {
      const el = document.querySelector(`[name="${n}"]`);
      if (el) el.style.border = "";
    });

    let firstInvalid = null;
    for (let fieldName of requiredFields) {
      const el = document.querySelector(`[name="${fieldName}"]`);
      if (!el) continue;
      const val = el.value;
      if (val === null || val === undefined || String(val).trim() === "") {
        firstInvalid = el;
        el.style.border = "2px solid #ff4d4f";
        break;
      }
    }

    // image check
    if (!imageUpload.files || imageUpload.files.length === 0) {
      if (!firstInvalid) {
        alert("⚠️ Please upload or capture a fundus image before submitting.");
      }
      e.preventDefault();
      return false;
    }

    if (firstInvalid) {
      e.preventDefault();
      firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
      firstInvalid.focus();
      return false;
    }

    // Success feedback (non-blocking)
    // Note: avoid alert if you prefer UX without modal
    // alert("✅ All details entered! Generating your reports...");
    return true;
  });
}
