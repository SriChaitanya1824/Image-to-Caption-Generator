from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import uvicorn, io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Image-to-Caption Generator</title>
<style>
body { font-family: sans-serif; text-align: center; padding: 50px; background: #f9fafb; }
h1 { font-size: 2em; margin-bottom: 20px; }
input[type=file] { margin: 20px; }
#caption { margin-top: 20px; font-size: 1.2em; }
</style>
</head>
<body>
<h1>🖼️ Image-to-Caption Generator</h1>
<input type="file" id="fileInput" accept="image/*">
<div id="caption"></div>
<script>
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('caption').innerText = 'Generating caption...';
  const formData = new FormData(); formData.append('file', file);
  const res = await fetch('/generate-caption', { method: 'POST', body: formData });
  const data = await res.json();
  document.getElementById('caption').innerText = data.caption;
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return html_content

@app.post("/generate-caption")
async def generate_caption(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"caption": caption}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
