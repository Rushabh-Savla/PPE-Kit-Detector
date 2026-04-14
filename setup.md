# PPE Detector - Setup Guide

A comprehensive full-stack safety management system with PPE detection, face recognition, and attendance tracking for mine safety operations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Database Setup](#database-setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Default Credentials](#default-credentials)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.10 - 3.13 | Python 3.14 not yet supported |
| Node.js | 18+ | LTS version recommended |
| MongoDB | 6.0+ | Local or Atlas (cloud) |
| Git | Latest | For cloning the repository |
| CUDA (Optional) | 11.8+ | For GPU-accelerated inference |

---

## Quick Start

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd ppe-detector

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup (new terminal)
cd nextjs-frontend
npm install

# Start MongoDB (ensure it's running)
# Then start the application
# Terminal 1: Backend
cd backend && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd nextjs-frontend && npm run dev
```

Access the application:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Detailed Setup

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate (macOS/Linux)
   source venv/bin/activate

   # Activate (Windows)
   venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt

   # Required for WebSocket support (video streaming)
   pip install 'uvicorn[standard]'
   # Or alternatively:
   pip install websockets
   ```

4. **Create environment file**
   ```bash
   # Create .env file in backend directory
   touch .env
   ```

   Add the following to `.env`:
   ```env
   MONGODB_URI=mongodb://localhost:27017/sih_safety_system
   JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
   HOST=0.0.0.0
   PORT=8000
   CORS_ORIGINS=http://localhost:3000
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd nextjs-frontend
   ```

2. **Install Node dependencies**
   ```bash
   npm install
   ```

3. **Create environment file**
   ```bash
   # Create .env.local file
   touch .env.local
   ```

   Add the following to `.env.local`:
   ```env
   MONGODB_URI=mongodb://localhost:27017/sih_safety_system
   NEXT_PUBLIC_API_URL=http://localhost:8000
   JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
   ```

### Database Setup

#### Option A: Local MongoDB

1. **Install MongoDB Community Edition**
   - [Download MongoDB](https://www.mongodb.com/try/download/community)

2. **Start MongoDB service**
   ```bash
   # macOS (Homebrew)
   brew services start mongodb-community

   # Linux
   sudo systemctl start mongod

   # Windows
   net start MongoDB
   ```

3. **Verify connection**
   ```bash
   mongosh
   # Should connect to mongodb://localhost:27017
   ```

#### Option B: MongoDB Atlas (Cloud)

1. Create a free account at [mongodb.com](https://www.mongodb.com/atlas)
2. Create a new cluster
3. Create a database user
4. Get the connection string
5. Update `MONGODB_URI` in your `.env` files with the Atlas connection string

#### Seed Initial Data (Optional)

```bash
cd backend
source venv/bin/activate
python seed_data.py
# or for comprehensive test data
python seed_comprehensive.py
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/sih_safety_system` |
| `MONGO_ATLAS_URI` | MongoDB Atlas connection (optional) | - |
| `JWT_SECRET` | Secret key for JWT tokens | **Must be changed in production** |
| `HOST` | Backend host address | `0.0.0.0` |
| `PORT` | Backend port | `8000` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:3000` |

### Frontend (`nextjs-frontend/.env.local`)

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/sih_safety_system` |
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |
| `JWT_SECRET` | Secret key for JWT tokens | Must match backend |

### Gas Sensor Bridge (`backend/gas_bridge.py`)

Configure these variables in the file if using gas sensor integration:

| Variable | Description |
|----------|-------------|
| `MINE_ID` | Your mine ID from MongoDB |
| `SENSOR_ID` | Gas sensor identifier |
| `USERNAME` | API authentication username |
| `PASSWORD` | API authentication password |

---

## Running the Application

### Development Mode

Open **three terminals**:

**Terminal 1 - Backend API**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend**
```bash
cd nextjs-frontend
npm run dev
```

**Terminal 3 - Gas Sensor Bridge (Optional)**
```bash
cd backend
source venv/bin/activate
python gas_bridge.py
```

### Production Build

**Frontend**
```bash
cd nextjs-frontend
npm run build
npm start
```

**Backend**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Default Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Super Admin | `superadmin` | `admin123` |

> **Important**: Change these credentials immediately after first login!

---

## Project Structure

```
ppe-detector/
├── backend/                    # Python FastAPI backend
│   ├── main.py                # Main application entry
│   ├── requirements.txt       # Python dependencies
│   ├── database.py            # MongoDB connection
│   ├── auth.py                # JWT authentication
│   ├── detector.py            # ML detection module
│   ├── routes/                # API route handlers
│   │   ├── auth.py
│   │   ├── workers.py
│   │   ├── gates.py
│   │   ├── alerts.py
│   │   └── ...
│   └── ml/                    # Machine learning modules
│       ├── models.py
│       ├── training.py
│       └── prediction_service.py
│
├── nextjs-frontend/           # Next.js 16 frontend
│   ├── package.json
│   ├── src/
│   │   ├── app/              # Next.js pages
│   │   │   ├── login/
│   │   │   ├── ppe-detection/
│   │   │   ├── gate-monitoring/
│   │   │   ├── attendance/
│   │   │   └── ...
│   │   └── components/       # React components
│   └── .env.local            # Frontend environment
│
├── models/                    # ML model configurations
│   └── ppe_detector/
│       └── args.yaml
│
├── train_ppe.py              # PPE model training script
├── yolov8m.pt                # Pre-trained YOLOv8 model
└── setup.md                  # This file
```

---

## Technologies Used

### Backend
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **MongoDB + Motor** - Database with async driver
- **PyTorch + Ultralytics** - YOLOv8 for PPE detection
- **FaceNet-PyTorch** - Face recognition
- **OpenCV** - Image processing
- **python-jose** - JWT authentication
- **APScheduler** - Job scheduling

### Frontend
- **Next.js 16** - React framework
- **React 19** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS 4** - Styling
- **Zustand** - State management
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **React-Webcam** - Camera access

### ML Models
- **YOLOv8** - PPE detection (helmet, vest, gloves, goggles, mask, safety shoes)
- **FaceNet** - Face recognition embeddings

---

## Troubleshooting

### Backend Issues

**ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**MongoDB connection failed**
```bash
# Verify MongoDB is running
mongosh
# Check MONGODB_URI in .env
```

**Port already in use**
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>
```

**WebSocket connection failed / "No supported WebSocket library detected"**
```bash
# Install WebSocket support for uvicorn
pip install 'uvicorn[standard]'
# Or alternatively:
pip install websockets

# Restart the backend server after installation
```
This error appears as:
- `WARNING: No supported WebSocket library detected`
- `WARNING: Unsupported upgrade request`
- WebSocket connections returning 404

### Frontend Issues

**npm install fails**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**API connection error**
- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify CORS settings in backend

**TypeError: "X is not a function" errors**

If you see errors like:
- `mineApi.list is not a function`
- `hasMinRole is not a function`
- `videoApi.start is not a function`

These indicate mismatches between frontend API calls and the actual API methods. The API methods available are:
- `mineApi.getAll()` (not `list`)
- `workerApi.getAll(params)` (not `list`)
- `userApi.getAll(params)` (not `list`)
- `alertApi.getAll(params)` (not `list`)
- `videoApi.start(source, maxFps)` for video streaming

**Export not found errors**

If you see `Export X doesn't exist in target module`:
- Check if the function is exported from the source file
- Verify the import path is correct
- Run `npm run build` to see all TypeScript errors

### Camera/Video Streaming Issues

**Camera not accessible**
- Grant browser camera permissions
- Close other apps using the camera
- Check browser console for errors

**Video WebSocket connection fails**
1. Ensure `websockets` or `uvicorn[standard]` is installed
2. Check the WebSocket URL matches the backend endpoint (`/ws/video`)
3. Verify the backend is running and accessible
4. Check browser console for specific error messages

**"Video stream already running" message**
- This is normal if the stream was started previously
- The backend maintains stream state between requests

**Face not recognized**
- Ensure good lighting conditions
- Face should be clearly visible
- Re-register face if needed

**Slow detection**
- Enable GPU acceleration (requires CUDA)
- Reduce image resolution
- Use YOLOv8n instead of YOLOv8m for faster inference

### First Run Notes

- First run downloads ~200MB of ML models from HuggingFace
- Face registration is required before face recognition works
- Database seeding is recommended for testing
- WebSocket support requires `uvicorn[standard]` or `websockets` package

---

## Support

For issues and feature requests, please open an issue in the repository.

## License

[Add your license information here]
