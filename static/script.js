const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
    });
});

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function setupUpload(config) {
    const dropZone = document.getElementById(config.dropId);
    const input = document.getElementById(config.inputId);
    const preview = document.getElementById(config.previewId);
    const nameEl = document.getElementById(config.nameId);
    const sizeEl = document.getElementById(config.sizeId);
    const removeBtn = document.getElementById(config.removeId);
    const btn = document.getElementById(config.btnId);
    const errorEl = document.getElementById(config.errorId);
    const form = document.getElementById(config.formId);

    function handleFile(file) {
        if (!file) return;
        const ext = file.name.split('.').pop().toLowerCase();
        if (!config.allowedExt.includes(ext)) {
            errorEl.classList.add('active');
            return;
        }
        errorEl.classList.remove('active');

        nameEl.textContent = file.name;
        sizeEl.textContent = formatSize(file.size);
        preview.classList.add('active');
        dropZone.style.display = 'none';
        btn.disabled = false;

        if (config.type === 'image') {
            const thumb = document.getElementById(config.thumbId);
            thumb.src = URL.createObjectURL(file);
        }
        if (config.type === 'video') {
            const thumb = document.getElementById(config.thumbId);
            thumb.src = URL.createObjectURL(file);
        }
        if (config.type === 'audio') {
            const player = document.getElementById(config.playerId);
            player.src = URL.createObjectURL(file);
        }
    }

    dropZone.addEventListener('click', () => input.click());
    input.addEventListener('change', () => handleFile(input.files[0]));

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            handleFile(file);
        }
    });

    removeBtn.addEventListener('click', () => {
        input.value = '';
        preview.classList.remove('active');
        dropZone.style.display = 'block';
        btn.disabled = true;
    });

    form.addEventListener('submit', (e) => {
        if (!input.files.length) {
            e.preventDefault();
            errorEl.classList.add('active');
            return;
        }
        showLoading(config.loadingText);
    });
}

setupUpload({
    dropId: 'drop-image', inputId: 'input-image', previewId: 'preview-image',
    nameId: 'preview-image-name', sizeId: 'preview-image-size', thumbId: 'preview-image-thumb',
    removeId: 'remove-image', btnId: 'btn-image', errorId: 'image-error', formId: 'panel-image',
    allowedExt: ['jpg', 'jpeg', 'png', 'webp'], type: 'image', loadingText: 'Analyzing image...'
});

setupUpload({
    dropId: 'drop-audio', inputId: 'input-audio', previewId: 'preview-audio',
    nameId: 'preview-audio-name', sizeId: 'preview-audio-size', playerId: 'preview-audio-player',
    removeId: 'remove-audio', btnId: 'btn-audio', errorId: 'audio-error', formId: 'panel-audio',
    allowedExt: ['mp3', 'wav', 'flac'], type: 'audio', loadingText: 'Analyzing audio...'
});

setupUpload({
    dropId: 'drop-video', inputId: 'input-video', previewId: 'preview-video',
    nameId: 'preview-video-name', sizeId: 'preview-video-size', thumbId: 'preview-video-thumb',
    removeId: 'remove-video', btnId: 'btn-video', errorId: 'video-error', formId: 'panel-video',
    allowedExt: ['mp4', 'mov', 'avi'], type: 'video', loadingText: 'Analyzing video...'
});

function showLoading(text) {
    document.getElementById('loading-text').textContent = text || 'Analyzing content...';
    document.getElementById('loading-overlay').classList.add('active');
}