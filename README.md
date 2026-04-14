# Safety System - PPE Detection & Attendance

A comprehensive safety management system with face recognition-based attendance and PPE (Personal Protective Equipment) detection using computer vision.

## Features

- **Face Recognition Attendance**: Check-in/check-out using webcam face recognition
- **PPE Detection**: Detects helmet, vest, gloves, goggles, mask, and safety shoes
- **Violation Logging**: Automatically logs PPE violations with timestamps and images
- **Dashboard**: Real-time overview of attendance and safety compliance
- **Reports**: Generate attendance and violation reports with CSV export
- **Employee Management**: Register employees with face recognition

## Tech Stack

### Backend
- **FastAPI**: Python web framework
- **MongoDB**: Database for storing attendance, violations, and employee data
- **YOLOv8**: PPE detection models
- **FaceNet**: Face recognition with MTCNN

### Frontend
- **Next.js 16**: React framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS
- **Zustand**: State management

## Project Structure

```
sih-final/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── detector.py      # ML detection module
│   ├── database.py      # MongoDB connection
│   ├── auth.py          # JWT authentication
│   ├── schemas.py       # Pydantic models
│   ├── requirements.txt # Python dependencies
│   └── .env             # Backend environment variables
├── nextjs-frontend/
│   ├── src/
│   │   ├── app/         # Next.js pages
│   │   ├── components/  # React components
│   │   ├── lib/         # API utilities and store
│   │   └── types/       # TypeScript types
│   ├── .env.local       # Frontend environment variables
│   └── package.json
├── frontend/            # Legacy HTML frontend
└── README.md
```

## Models Used

1. **PPE Detection Model 1**: [keremberke/yolov8m-protective-equipment-detection](https://huggingface.co/keremberke/yolov8m-protective-equipment-detection)
   - Detects: helmet, goggles, glove, mask, shoes
   - Also detects missing: no_helmet, no_goggles, no_glove, no_mask, no_shoes

2. **PPE Detection Model 2**: [Tanishjain9/yolov8n-ppe-detection-6classes](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes)
   - Detects: Helmet, Vest, Goggles, Gloves, Mask, Safety Shoe

3. **Face Detection**: [arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

4. **Face Recognition**: FaceNet (via facenet-pytorch)

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas)
- CUDA-compatible GPU (recommended for faster ML inference)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Edit .env with your MongoDB connection string
```

### 2. Frontend Setup

```bash
cd nextjs-frontend

# Install dependencies
npm install

# Configure environment
# Edit .env.local with your settings
```

### 3. MongoDB Setup

Make sure MongoDB is running locally or update the `MONGODB_URI` in both `.env` files to point to your MongoDB instance.

Default connection: `mongodb://localhost:27017/sih_safety_system`

### 4. Environment Variables

**Backend (.env)**
```env
MONGODB_URI=mongodb://localhost:27017/sih_safety_system
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000
```

**Frontend (.env.local)**
```env
MONGODB_URI=mongodb://localhost:27017/sih_safety_system
NEXT_PUBLIC_API_URL=http://localhost:8000
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

## Running the Application

### Start Backend

```bash
cd backend
python main.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Start Frontend

```bash
cd nextjs-frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Default Login

- **Username**: admin
- **Password**: admin123

Change these credentials after first login!

## API Endpoints

### Authentication
- `POST /auth/login` - Admin login
- `GET /auth/verify` - Verify token

### Employees
- `GET /employees` - List employees
- `POST /employees` - Create employee
- `PUT /employees/{id}` - Update employee
- `DELETE /employees/{id}` - Delete employee
- `POST /employees/{id}/register-face` - Register face

### Attendance
- `POST /attendance/check-in` - Check-in with face
- `POST /attendance/check-out` - Check-out with face
- `GET /attendance` - Get attendance records
- `GET /attendance/today` - Get today's attendance

### Detection
- `POST /detect` - Detect PPE and faces
- `POST /detect-and-log` - Detect and log violations

### Violations
- `GET /violations` - Get violation records
- `GET /violations/today` - Get today's violations

### Dashboard & Reports
- `GET /dashboard/stats` - Dashboard statistics
- `GET /reports/attendance` - Attendance report
- `GET /reports/violations` - Violations report

## PPE Detection Labels

| Label | Color | Meaning |
|-------|-------|---------|
| Helmet, Vest, Goggles, etc. | Green | PPE detected (worn) |
| NO Helmet, NO Goggles, etc. | Red | Missing PPE (violation) |
| Face | Yellow | Detected face (with name if recognized) |

## Camera Support

The application supports:
- **Webcam**: Real-time capture using browser's media API (Logitech C720 supported)
- **File Upload**: Upload images as a fallback

## Troubleshooting

### Camera not working
1. Check browser permissions for camera access
2. Ensure no other application is using the camera
3. Try refreshing the page

### Face not recognized
1. Ensure good lighting
2. Face should be clearly visible
3. Register face again if recognition fails

### Backend errors
1. Check MongoDB connection
2. Verify all dependencies are installed
3. Check GPU drivers if using CUDA

## Notes

- First run will download models from HuggingFace (~200MB)
- Face recognition requires registering faces first
- GPU (CUDA) is used if available for faster inference
- Use Python 3.10+ (Python 3.14 is not yet supported by all packages)

## License

MIT License
