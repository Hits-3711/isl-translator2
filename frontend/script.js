const video = document.getElementById('webcam');
const resultBox = document.getElementById('result-box');

// Start Webcam Stream
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Webcam error: ", err);
        resultBox.innerText = "Webcam Access Denied!";
    });

// Canvas to capture frames
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;
const ctx = canvas.getContext('2d');

// Send frame to backend every 100ms (~10 FPS)
setInterval(() => {
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    resultBox.innerText = `Letter: ${data.prediction} (${data.confidence.toFixed(0)}%)`;
                    resultBox.style.background = '#28a745'; // Green
                } else if (data.status === 'analyzing') {
                    resultBox.innerText = "Analyzing Sign...";
                    resultBox.style.background = '#ffc107'; // Yellow/Orange
                } else {
                    resultBox.innerText = "No Hands Detected";
                    resultBox.style.background = '#dc3545'; // Red
                }
            })
            .catch(err => console.error('Prediction error:', err));
        }, 'image/jpeg', 0.7);
    }
}, 100);