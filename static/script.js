document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.querySelector('.preview-container');
    const analyzeBtn = document.querySelector('.analyze-btn');
    const resultPlaceholder = document.querySelector('.result-placeholder');
    const loader = document.querySelector('.loader');
    const resultContent = document.querySelector('.result-content');
    const predictionLabel = document.querySelector('.prediction-label');
    const confidenceBar = document.querySelector('.confidence-bar');
    const confidenceValue = document.querySelector('.confidence-value');

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    previewContainer.style.display = 'block';
                    analyzeBtn.classList.add('active');
                    resetResult();
                };
                reader.readAsDataURL(file);
            } else {
                alert('Please upload a valid image file.');
            }
        }
    }

    function resetResult() {
        resultContent.style.display = 'none';
        resultPlaceholder.style.display = 'block';
        loader.style.display = 'none';
    }

    analyzeBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        // UI State: Loading
        analyzeBtn.classList.remove('active');
        resultPlaceholder.style.display = 'none';
        loader.style.display = 'block';
        resultContent.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            // UI State: Done
            loader.style.display = 'none';
            resultContent.style.display = 'block';

            if (data.error) {
                predictionLabel.textContent = "Error";
                predictionLabel.className = "prediction-label";
                confidenceValue.textContent = data.error;
                confidenceBar.style.width = '0%';
            } else {
                predictionLabel.textContent = data.label;
                const isStroke = data.label === "Stroke";
                predictionLabel.className = `prediction-label ${isStroke ? 'stroke' : 'normal'}`;
                
                confidenceValue.textContent = data.confidence;
                // Parse percentage string to number for width
                const width = parseFloat(data.confidence);
                setTimeout(() => {
                    confidenceBar.style.width = width + '%';
                    confidenceBar.style.backgroundColor = isStroke ? '#ff4b4b' : '#00d26a';
                }, 100);
            }

        } catch (error) {
            console.error('Error:', error);
            loader.style.display = 'none';
            resultPlaceholder.style.display = 'block';
            resultPlaceholder.textContent = "An error occurred. Please try again.";
        }
    });
});
