const express = require('express');
const multer = require('multer');
const { createWorker } = require('tesseract.js');
const cors = require('cors');
const fs = require('fs');

const app = express();
app.use(express.json());
app.use(cors());

// Temporary storage for uploaded images
const upload = multer({ dest: 'uploads/' });

// Health Check Endpoint (Cron-job အတွက်)
app.get('/', (req, res) => {
    res.status(200).json({ status: 'OCR Server is active and running!' });
});

app.get('/health', (req, res) => {
    res.status(200).json({ status: 'OK' });
});

// OCR Processing Endpoint
app.post('/api/ocr/process', upload.single('image'), async (req, res) => {
    const file = req.file;
    if (!file) {
        return res.status(400).json({ error: 'No image provided' });
    }

    try {
        // Initialize Tesseract worker
        const worker = await createWorker('eng');
        const ret = await worker.recognize(file.path);
        await worker.terminate();

        // Clean up uploaded temp file from server storage
        if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
        }

        res.status(200).json({ 
            success: true, 
            extractedText: ret.data.text 
        });
    } catch (error) {
        // Clean up file if error occurs
        if (file && fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
        }
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`OCR Server running on port ${PORT}`));
