'use client';

import { useRef, useCallback, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import {
  Camera as CameraIcon,
  Upload,
  RefreshCw,
  CheckCircle,
  ArrowLeft,
  ArrowRight,
  Circle,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { dataURLtoFile } from '@/lib/utils';

// MediaPipe imports
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
  FaceLandmarkerResult,
} from '@mediapipe/tasks-vision';

type ScanStep = 'center' | 'left' | 'right' | 'complete';

interface CapturedImages {
  center: string | null;
  left: string | null;
  right: string | null;
}

interface FaceScanCaptureProps {
  onCapture: (files: File[]) => void;
  disabled?: boolean;
  onCancel?: () => void;
}

// Face mesh tessellation indices for drawing the grid overlay
const FACE_MESH_TESSELATION = [
  // Simplified key connections for visual effect
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454],
  [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400],
  [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
  [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54],
  [54, 103], [103, 67], [67, 109], [109, 10],
  // Nose
  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2],
  // Left eye
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
  [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161], [161, 246], [246, 33],
  // Right eye
  [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249], [249, 263],
  [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362],
  // Lips outer
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321],
  [321, 375], [375, 291], [291, 409], [409, 270], [270, 269], [269, 267], [267, 0], [0, 37],
  [37, 39], [39, 40], [40, 185], [185, 61],
  // Jawline
  [234, 93], [93, 132], [132, 58], [58, 172], [172, 136], [136, 150], [150, 149], [149, 176],
  [176, 148], [148, 152], [152, 377], [377, 400], [400, 378], [378, 379], [379, 365], [365, 397],
  [397, 288], [288, 361], [361, 323], [323, 454],
];

// Key landmark indices for head pose estimation
const LANDMARK_INDICES = {
  NOSE_TIP: 1,
  CHIN: 152,
  LEFT_EYE_OUTER: 33,
  RIGHT_EYE_OUTER: 263,
  LEFT_EYE_INNER: 133,
  RIGHT_EYE_INNER: 362,
  LEFT_MOUTH: 61,
  RIGHT_MOUTH: 291,
  FOREHEAD: 10,
};

export default function FaceScanCapture({ onCapture, disabled = false, onCancel }: FaceScanCaptureProps) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [currentStep, setCurrentStep] = useState<ScanStep>('center');
  const [capturedImages, setCapturedImages] = useState<CapturedImages>({
    center: null,
    left: null,
    right: null,
  });
  const [useCamera, setUseCamera] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [headPose, setHeadPose] = useState<{ yaw: number; pitch: number; roll: number } | null>(null);
  const [poseCorrect, setPoseCorrect] = useState(false);
  const [autoCapturing, setAutoCapturing] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  // Initialize MediaPipe FaceLandmarker
  useEffect(() => {
    const initFaceLandmarker = async () => {
      try {
        setIsLoading(true);
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
          runningMode: 'VIDEO',
          numFaces: 1,
        });

        faceLandmarkerRef.current = faceLandmarker;
        setIsLoading(false);
      } catch (error) {
        console.error('Error initializing FaceLandmarker:', error);
        setIsLoading(false);
        setCameraError('Failed to load face detection model. Please refresh and try again.');
      }
    };

    initFaceLandmarker();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close();
      }
    };
  }, []);

  // Calculate head pose from face landmarks
  const calculateHeadPose = useCallback((landmarks: { x: number; y: number; z: number }[]) => {
    // Get key landmark positions
    const noseTip = landmarks[LANDMARK_INDICES.NOSE_TIP];
    const chin = landmarks[LANDMARK_INDICES.CHIN];
    const leftEyeOuter = landmarks[LANDMARK_INDICES.LEFT_EYE_OUTER];
    const rightEyeOuter = landmarks[LANDMARK_INDICES.RIGHT_EYE_OUTER];
    const forehead = landmarks[LANDMARK_INDICES.FOREHEAD];

    // Calculate yaw (left-right rotation) using eye positions and nose tip
    const eyeMidpointX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const noseOffsetX = noseTip.x - eyeMidpointX;
    const eyeDistance = Math.abs(rightEyeOuter.x - leftEyeOuter.x);
    const yaw = (noseOffsetX / eyeDistance) * 90; // Approximate yaw angle in degrees

    // Calculate pitch (up-down rotation) using forehead, nose, and chin
    const faceHeight = Math.abs(chin.y - forehead.y);
    const noseRelativeY = (noseTip.y - forehead.y) / faceHeight;
    const pitch = (noseRelativeY - 0.6) * 90; // Approximate pitch angle

    // Calculate roll (tilt) using eye positions
    const eyeDeltaY = rightEyeOuter.y - leftEyeOuter.y;
    const roll = Math.atan2(eyeDeltaY, eyeDistance) * (180 / Math.PI);

    return { yaw, pitch, roll };
  }, []);

  // Check if pose matches the required position for current step
  const checkPoseForStep = useCallback((pose: { yaw: number; pitch: number; roll: number }, step: ScanStep): boolean => {
    const { yaw, pitch, roll } = pose;

    // Ensure face is not tilted too much
    if (Math.abs(roll) > 15) return false;
    // Ensure face is not looking up/down too much
    if (Math.abs(pitch) > 20) return false;

    switch (step) {
      case 'center':
        return Math.abs(yaw) < 10; // Face looking straight
      case 'left':
        return yaw > 20 && yaw < 45; // Face turned left (from viewer's perspective)
      case 'right':
        return yaw < -20 && yaw > -45; // Face turned right
      default:
        return false;
    }
  }, []);

  // Draw face mesh grid overlay
  const drawFaceMesh = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: { x: number; y: number; z: number }[],
    width: number,
    height: number,
    isCorrectPose: boolean
  ) => {
    ctx.clearRect(0, 0, width, height);

    // Draw grid overlay with glow effect
    const gridColor = isCorrectPose ? 'rgba(34, 197, 94, 0.8)' : 'rgba(249, 115, 22, 0.8)';
    const glowColor = isCorrectPose ? 'rgba(34, 197, 94, 0.3)' : 'rgba(249, 115, 22, 0.3)';

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1.5;
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 8;

    // Draw tessellation lines
    FACE_MESH_TESSELATION.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];

      if (startPoint && endPoint) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.stroke();
      }
    });

    // Draw key landmark points
    ctx.shadowBlur = 0;
    const keyLandmarks = [
      LANDMARK_INDICES.NOSE_TIP,
      LANDMARK_INDICES.LEFT_EYE_OUTER,
      LANDMARK_INDICES.RIGHT_EYE_OUTER,
      LANDMARK_INDICES.LEFT_EYE_INNER,
      LANDMARK_INDICES.RIGHT_EYE_INNER,
      LANDMARK_INDICES.LEFT_MOUTH,
      LANDMARK_INDICES.RIGHT_MOUTH,
      LANDMARK_INDICES.CHIN,
      LANDMARK_INDICES.FOREHEAD,
    ];

    ctx.fillStyle = isCorrectPose ? 'rgba(34, 197, 94, 1)' : 'rgba(249, 115, 22, 1)';
    keyLandmarks.forEach((idx) => {
      const point = landmarks[idx];
      if (point) {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw face bounding oval
    const centerX = (landmarks[LANDMARK_INDICES.LEFT_EYE_OUTER].x + landmarks[LANDMARK_INDICES.RIGHT_EYE_OUTER].x) / 2 * width;
    const centerY = (landmarks[LANDMARK_INDICES.FOREHEAD].y + landmarks[LANDMARK_INDICES.CHIN].y) / 2 * height;
    const radiusX = Math.abs(landmarks[LANDMARK_INDICES.RIGHT_EYE_OUTER].x - landmarks[LANDMARK_INDICES.LEFT_EYE_OUTER].x) * width * 1.3;
    const radiusY = Math.abs(landmarks[LANDMARK_INDICES.CHIN].y - landmarks[LANDMARK_INDICES.FOREHEAD].y) * height * 0.8;

    ctx.strokeStyle = isCorrectPose ? 'rgba(34, 197, 94, 0.5)' : 'rgba(249, 115, 22, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }, []);

  // Main detection loop
  const detectFace = useCallback(async () => {
    if (
      !webcamRef.current?.video ||
      !canvasRef.current ||
      !faceLandmarkerRef.current ||
      webcamRef.current.video.readyState !== 4 ||
      currentStep === 'complete'
    ) {
      animationFrameRef.current = requestAnimationFrame(detectFace);
      return;
    }

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      animationFrameRef.current = requestAnimationFrame(detectFace);
      return;
    }

    // Ensure canvas matches video dimensions
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    try {
      const startTimeMs = performance.now();
      const results = faceLandmarkerRef.current.detectForVideo(video, startTimeMs);

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        setFaceDetected(true);

        // Calculate head pose
        const pose = calculateHeadPose(landmarks);
        setHeadPose(pose);

        // Check if pose matches current step requirement
        const isPoseCorrect = checkPoseForStep(pose, currentStep);
        setPoseCorrect(isPoseCorrect);

        // Draw the face mesh overlay
        drawFaceMesh(ctx, landmarks, canvas.width, canvas.height, isPoseCorrect);

      } else {
        setFaceDetected(false);
        setHeadPose(null);
        setPoseCorrect(false);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    } catch (error) {
      console.error('Error detecting face:', error);
    }

    animationFrameRef.current = requestAnimationFrame(detectFace);
  }, [currentStep, calculateHeadPose, checkPoseForStep, drawFaceMesh]);

  // Start detection when camera is ready
  useEffect(() => {
    if (!isLoading && useCamera && !cameraError) {
      animationFrameRef.current = requestAnimationFrame(detectFace);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isLoading, useCamera, cameraError, detectFace]);

  // Auto-capture with countdown when pose is correct
  useEffect(() => {
    let countdownTimer: NodeJS.Timeout | null = null;

    if (poseCorrect && faceDetected && !autoCapturing && useCamera && currentStep !== 'complete') {
      setAutoCapturing(true);
      setCountdown(3);

      countdownTimer = setInterval(() => {
        setCountdown((prev) => {
          if (prev === null || prev <= 1) {
            clearInterval(countdownTimer!);
            // Capture the image
            captureCurrentStep();
            return null;
          }
          return prev - 1;
        });
      }, 1000);
    } else if (!poseCorrect || !faceDetected) {
      setAutoCapturing(false);
      setCountdown(null);
      if (countdownTimer) {
        clearInterval(countdownTimer);
      }
    }

    return () => {
      if (countdownTimer) {
        clearInterval(countdownTimer);
      }
    };
  }, [poseCorrect, faceDetected, autoCapturing, useCamera, currentStep]);

  // Capture current step image
  const captureCurrentStep = useCallback(() => {
    if (!webcamRef.current || currentStep === 'complete') return;

    const imageSrc = webcamRef.current.getScreenshot({
      width: 1280,
      height: 720,
    });

    if (imageSrc) {
      setCapturedImages((prev) => ({
        ...prev,
        [currentStep]: imageSrc,
      }));

      // Move to next step
      if (currentStep === 'center') {
        setCurrentStep('left');
      } else if (currentStep === 'left') {
        setCurrentStep('right');
      } else if (currentStep === 'right') {
        setCurrentStep('complete');
      }

      setAutoCapturing(false);
      setCountdown(null);
    }
  }, [currentStep]);

  // Handle manual capture button
  const handleManualCapture = useCallback(() => {
    if (faceDetected && currentStep !== 'complete') {
      captureCurrentStep();
    }
  }, [faceDetected, currentStep, captureCurrentStep]);

  // Handle file upload (fallback)
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setCapturedImages({
          center: result,
          left: null,
          right: null,
        });
        setCurrentStep('complete');
      };
      reader.readAsDataURL(file);
    }
  };

  // Reset all captures
  const resetCapture = () => {
    setCapturedImages({ center: null, left: null, right: null });
    setCurrentStep('center');
    setAutoCapturing(false);
    setCountdown(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Submit captured images
  const handleSubmit = useCallback(() => {
    const files: File[] = [];

    if (capturedImages.center) {
      files.push(dataURLtoFile(capturedImages.center, 'face_center.png'));
    }
    if (capturedImages.left) {
      files.push(dataURLtoFile(capturedImages.left, 'face_left.png'));
    }
    if (capturedImages.right) {
      files.push(dataURLtoFile(capturedImages.right, 'face_right.png'));
    }

    if (files.length > 0) {
      onCapture(files);
    }
  }, [capturedImages, onCapture]);

  const handleCameraError = () => {
    setCameraError('Unable to access camera. Please check permissions or use file upload.');
    setUseCamera(false);
  };

  // Get step instructions
  const getStepInstructions = () => {
    switch (currentStep) {
      case 'center':
        return {
          title: 'Look Straight',
          description: 'Position your face in the center and look directly at the camera',
          icon: <Circle className="text-orange-500" size={24} />,
        };
      case 'left':
        return {
          title: 'Turn Left',
          description: 'Slowly turn your head to the left (your left)',
          icon: <ArrowLeft className="text-orange-500" size={24} />,
        };
      case 'right':
        return {
          title: 'Turn Right',
          description: 'Slowly turn your head to the right (your right)',
          icon: <ArrowRight className="text-orange-500" size={24} />,
        };
      case 'complete':
        return {
          title: 'Scan Complete',
          description: 'Your face has been scanned from all angles',
          icon: <CheckCircle className="text-green-500" size={24} />,
        };
    }
  };

  const instructions = getStepInstructions();

  return (
    <div className="space-y-4">
      {/* Toggle buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => {
            setUseCamera(true);
            setCameraError(null);
            resetCapture();
          }}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
            useCamera
              ? 'bg-orange-600 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          <CameraIcon className="inline-block mr-2" size={18} />
          3D Face Scan
        </button>
        <button
          onClick={() => {
            setUseCamera(false);
            resetCapture();
          }}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
            !useCamera
              ? 'bg-orange-600 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          <Upload className="inline-block mr-2" size={18} />
          Upload Photo
        </button>
      </div>

      {/* Step indicator */}
      {useCamera && (
        <div className="bg-stone-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            {['center', 'left', 'right'].map((step, index) => {
              const isCompleted =
                (step === 'center' && capturedImages.center) ||
                (step === 'left' && capturedImages.left) ||
                (step === 'right' && capturedImages.right);
              const isCurrent = currentStep === step;

              return (
                <div key={step} className="flex items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                      isCompleted
                        ? 'bg-green-500 text-white'
                        : isCurrent
                        ? 'bg-orange-500 text-white'
                        : 'bg-stone-200 text-stone-500'
                    }`}
                  >
                    {isCompleted ? <CheckCircle size={16} /> : index + 1}
                  </div>
                  {index < 2 && (
                    <div
                      className={`w-16 sm:w-24 h-1 mx-2 rounded ${
                        isCompleted ? 'bg-green-500' : 'bg-stone-200'
                      }`}
                    />
                  )}
                </div>
              );
            })}
          </div>

          {/* Current step instructions */}
          <div className="flex items-center gap-3 bg-white rounded-lg p-3 border border-stone-200">
            {instructions.icon}
            <div>
              <h4 className="font-medium text-stone-800">{instructions.title}</h4>
              <p className="text-sm text-stone-500">{instructions.description}</p>
            </div>
          </div>
        </div>
      )}

      {/* Camera or Upload area */}
      <div className="relative aspect-video bg-stone-900 rounded-lg overflow-hidden">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-white">
            <Loader2 className="animate-spin mb-3" size={48} />
            <p>Loading face detection model...</p>
          </div>
        ) : currentStep === 'complete' ? (
          <div className="grid grid-cols-3 gap-2 p-2 h-full">
            {capturedImages.center && (
              <div className="relative rounded-lg overflow-hidden">
                <img
                  src={capturedImages.center}
                  alt="Center"
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                  Center
                </div>
              </div>
            )}
            {capturedImages.left && (
              <div className="relative rounded-lg overflow-hidden">
                <img
                  src={capturedImages.left}
                  alt="Left"
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                  Left
                </div>
              </div>
            )}
            {capturedImages.right && (
              <div className="relative rounded-lg overflow-hidden">
                <img
                  src={capturedImages.right}
                  alt="Right"
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                  Right
                </div>
              </div>
            )}
          </div>
        ) : useCamera ? (
          cameraError ? (
            <div className="flex items-center justify-center h-full text-white text-center p-4">
              <div>
                <AlertCircle className="mx-auto mb-3 text-red-400" size={48} />
                <p>{cameraError}</p>
              </div>
            </div>
          ) : (
            <>
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/png"
                mirrored={true}
                videoConstraints={{
                  width: 1280,
                  height: 720,
                  facingMode: 'user',
                }}
                onUserMediaError={handleCameraError}
                className="w-full h-full object-cover"
              />
              {/* Canvas overlay for face mesh */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ transform: 'scaleX(-1)' }}
              />

              {/* Countdown overlay */}
              {countdown !== null && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                  <div className="text-white text-8xl font-bold animate-pulse">
                    {countdown}
                  </div>
                </div>
              )}

              {/* Face detection status */}
              <div className="absolute top-4 left-4 right-4 flex justify-between items-start">
                <div
                  className={`px-3 py-2 rounded-lg text-sm font-medium ${
                    faceDetected
                      ? poseCorrect
                        ? 'bg-green-500 text-white'
                        : 'bg-orange-500 text-white'
                      : 'bg-red-500 text-white'
                  }`}
                >
                  {faceDetected
                    ? poseCorrect
                      ? 'Perfect! Hold still...'
                      : 'Adjust your position'
                    : 'No face detected'}
                </div>

                {headPose && (
                  <div className="bg-black/70 text-white text-xs px-3 py-2 rounded-lg">
                    <div>Yaw: {headPose.yaw.toFixed(1)}°</div>
                    <div>Pitch: {headPose.pitch.toFixed(1)}°</div>
                  </div>
                )}
              </div>

              {/* Face positioning guide */}
              <div className="absolute inset-0 pointer-events-none">
                <svg className="w-full h-full">
                  <ellipse
                    cx="50%"
                    cy="45%"
                    rx="20%"
                    ry="30%"
                    fill="none"
                    stroke={faceDetected ? (poseCorrect ? '#22c55e' : '#f97316') : '#ffffff'}
                    strokeWidth="2"
                    strokeDasharray="10,5"
                    opacity="0.5"
                  />
                </svg>
              </div>
            </>
          )
        ) : (
          <label className="flex flex-col items-center justify-center h-full cursor-pointer hover:bg-stone-800 transition-colors">
            <Upload size={48} className="text-stone-500 mb-2" />
            <span className="text-stone-400">Click to upload a face photo</span>
            <span className="text-stone-500 text-sm mt-1">(Single photo mode)</span>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        )}
      </div>

      {/* Captured thumbnails (for camera mode) */}
      {useCamera && currentStep !== 'complete' && (
        <div className="flex gap-2">
          {['center', 'left', 'right'].map((step) => {
            const image = capturedImages[step as keyof CapturedImages];
            return (
              <div
                key={step}
                className={`flex-1 aspect-video rounded-lg overflow-hidden border-2 ${
                  image ? 'border-green-500' : 'border-stone-200'
                }`}
              >
                {image ? (
                  <img src={image} alt={step} className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-stone-100 flex items-center justify-center text-stone-400 text-xs">
                    {step.charAt(0).toUpperCase() + step.slice(1)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        {useCamera && currentStep !== 'complete' && !cameraError && (
          <button
            onClick={handleManualCapture}
            disabled={disabled || !faceDetected}
            className="flex-1 py-3 px-4 bg-orange-600 text-white rounded-lg font-medium hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <CameraIcon className="inline-block mr-2" size={18} />
            Capture Manually
          </button>
        )}

        {(currentStep === 'complete' || (!useCamera && capturedImages.center)) && (
          <>
            <button
              onClick={resetCapture}
              disabled={disabled}
              className="flex-1 py-3 px-4 bg-stone-200 text-stone-700 rounded-lg font-medium hover:bg-stone-300 transition-colors disabled:opacity-50"
            >
              <RefreshCw className="inline-block mr-2" size={18} />
              Retake
            </button>
            <button
              onClick={handleSubmit}
              disabled={disabled}
              className="flex-1 py-3 px-4 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              <CheckCircle className="inline-block mr-2" size={18} />
              Register Face
            </button>
          </>
        )}
      </div>

      {/* Cancel button */}
      {onCancel && (
        <button
          onClick={onCancel}
          className="w-full py-2 text-stone-500 hover:text-stone-700 transition-colors"
        >
          Cancel
        </button>
      )}
    </div>
  );
}
