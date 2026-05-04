const video = document.getElementById('video');
const preview = document.getElementById('preview');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const fileInput = document.getElementById('fileInput');
const placeholder = document.getElementById('placeholder');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const result = document.getElementById('result');
const resultIcon = document.getElementById('resultIcon');
const resultText = document.getElementById('resultText');

let model = null;
let stream = null;
let mode = 'idle'; // idle | camera | preview

// ImageNet class 934 = "hotdog, hot dog, red hot"
const HOTDOG_CLASS_INDEX = 934;
const HOTDOG_THRESHOLD = 0.15;

async function loadModel() {
  if (model) return;
  loading.classList.add('show');
  loadingText.textContent = 'Loading model...';
  try {
    model = await tf.loadLayersModel('/model/model.json');
  } catch (e) {
    console.error(e);
    loadingText.textContent = 'Failed to load model. Please reload the page.';
    return;
  }
  loading.classList.remove('show');
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false
    });
    video.srcObject = stream;
    video.classList.add('active');
    preview.classList.remove('active');
    placeholder.style.display = 'none';
    mode = 'camera';
    captureBtn.textContent = '📸 Capture';
  } catch (err) {
    console.error('Camera error:', err);
    placeholder.textContent = 'Camera unavailable. Please use the gallery below.';
    placeholder.style.display = 'block';
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.classList.remove('active');
}

async function captureFromVideo() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataURL = canvas.toDataURL('image/jpeg', 0.9);
  preview.src = dataURL;
  preview.classList.add('active');
  video.classList.remove('active');
  stopCamera();
  mode = 'preview';
  await classifyImage(preview);
}

async function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (e) => {
    preview.src = e.target.result;
    preview.classList.add('active');
    video.classList.remove('active');
    placeholder.style.display = 'none';
    stopCamera();
    mode = 'preview';
    preview.onload = async () => {
      await classifyImage(preview);
    };
  };
  reader.readAsDataURL(file);
}

async function classifyImage(imgEl) {
  await loadModel();
  if (!model) return;
  loading.classList.add('show');
  loadingText.textContent = 'Analyzing...';

  // Wait for the image to fully load
  if (!imgEl.complete) {
    await new Promise(r => imgEl.onload = r);
  }

  try {
    const tensor = tf.browser.fromPixels(imgEl)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(tf.scalar(127.5))
      .sub(tf.scalar(1))
      .expandDims();

    const predictions = model.predict(tensor);
    const data = await predictions.data();
    tensor.dispose();
    predictions.dispose();

    const hotdogProb = data[HOTDOG_CLASS_INDEX];
    console.log('Hotdog probability:', hotdogProb);
    const isHotDog = hotdogProb > HOTDOG_THRESHOLD;

    loading.classList.remove('show');
    showResult(isHotDog);
  } catch (e) {
    console.error(e);
    loading.classList.remove('show');
    alert('Failed to analyze the image.');
  }
}

function showResult(isHotDog) {
  result.classList.remove('hotdog', 'nothotdog');
  if (isHotDog) {
    result.classList.add('hotdog');
    resultIcon.textContent = '✓';
    resultText.textContent = 'Hot Dog';
  } else {
    result.classList.add('nothotdog');
    resultIcon.textContent = '✕';
    resultText.textContent = 'Not Hot Dog';
  }
  result.classList.add('show');
}

function resetApp() {
  result.classList.remove('show');
  preview.classList.remove('active');
  preview.src = '';
  placeholder.style.display = 'block';
  placeholder.textContent = 'Tap "Take Photo" to start';
  captureBtn.textContent = '📷 Take Photo';
  mode = 'idle';
}

// Tap the result badge to reset and try again
result.addEventListener('click', resetApp);

captureBtn.addEventListener('click', async () => {
  if (mode === 'idle') {
    await startCamera();
  } else if (mode === 'camera') {
    await captureFromVideo();
  } else if (mode === 'preview') {
    resetApp();
    await startCamera();
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFile(e.target.files[0]);
  }
});

// Preload the model
window.addEventListener('load', () => {
  setTimeout(loadModel, 500);
});
