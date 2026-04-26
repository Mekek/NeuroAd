Версия системы с разделением на backend и frontend

распознавание лиц через Deepfake

---

## 🔧 Backend

```bash
cd C:\Code\NeuroAd\Implementation\Deepface_solution\backend
venv\Scripts\activate
uvicorn server:app --reload
```

---

## 🌐 Frontend (в новом терминале)

```bash
cd C:\Code\NeuroAd\Implementation\Deepface_solution\frontend
python -m http.server 8001
```

---

## 🌍 Открыть

```text
http://localhost:8001
```
