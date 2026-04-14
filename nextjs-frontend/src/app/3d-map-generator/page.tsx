'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import AppLayout from '@/components/AppLayout';
import { Upload, ChevronRight, Settings2, Play, ArrowLeft, Maximize2, Download, Flashlight, Map, User, Navigation, Eye, Radio, RotateCcw, Wifi, WifiOff } from 'lucide-react';
import Script from 'next/script';
import { useHelmetTracking, Position } from '@/hooks/useHelmetTracking';
import { useMineTrackingStore, MineData, WallData, RoomData } from '@/lib/store';

// Note: WallData, RoomData, MineData types are imported from store.ts

// Camera follow settings
const CAMERA_FOLLOW_OFFSET = { x: 0, y: 4, z: 8 }; // Behind and above worker
const CAMERA_LERP_FACTOR = 0.05; // Smooth camera follow

// Declare THREE and cv as global
declare global {
  interface Window {
    cv: any;
    THREE: any;
  }
}

export default function Map3DGeneratorPage() {
  // ==================== TRACKING STORE ====================
  const trackingStore = useMineTrackingStore();

  // ==================== STATE ====================
  const [cvReady, setCvReady] = useState(false);
  const [threeReady, setThreeReady] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<HTMLImageElement | null>(null);
  const [processedData, setProcessedData] = useState<MineData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [showViewer, setShowViewer] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showModeSelection, setShowModeSelection] = useState(false); // New: mode selection screen

  // Settings
  const [threshold, setThreshold] = useState(100);
  const [wallHeight, setWallHeight] = useState(4);
  const [scale, setScale] = useState(0.1);  // Much smaller scale - 10% of original
  const [addLights, setAddLights] = useState(true);
  const [addDust, setAddDust] = useState(true);

  // 2D Map and Simulation
  const [show2DMap, setShow2DMap] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const simulationRef = useRef<NodeJS.Timeout | null>(null);
  const map2DCanvasRef = useRef<HTMLCanvasElement>(null);

  // Refs
  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const processedCanvasRef = useRef<HTMLCanvasElement>(null);
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const minimapCanvasRef = useRef<HTMLCanvasElement>(null);

  // Three.js refs
  const sceneRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);
  const clockRef = useRef<any>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Worker tracking refs
  const workerMeshRef = useRef<any>(null);
  const entranceMarkerRef = useRef<any>(null);
  const targetCameraPositionRef = useRef({ x: 0, y: 4, z: 8 });
  const targetCameraLookAtRef = useRef({ x: 0, y: 1.7, z: 0 });

  // Movement state
  const moveStateRef = useRef({
    forward: false,
    backward: false,
    left: false,
    right: false,
    running: false
  });
  const eulerRef = useRef({ x: 0, y: 0 });
  const isPointerLockedRef = useRef(false);
  const flashlightRef = useRef<any>(null);
  const isFlashlightOnRef = useRef(true);

  // ==================== HELMET TRACKING HOOK ====================
  const trackingState = useHelmetTracking({
    enabled: trackingStore.isTracking,
    initialPosition: {
      x: trackingStore.entrancePosition.x,
      y: 1.7,
      z: trackingStore.entrancePosition.z,
    },
    onPositionUpdate: (position, heading) => {
      trackingStore.updateWorkerPosition(position, heading);
    },
    onConnectionChange: (connected, simulating) => {
      if (connected) {
        trackingStore.setConnectionStatus('connected');
      } else if (simulating) {
        trackingStore.setConnectionStatus('simulating');
      } else {
        trackingStore.setConnectionStatus('disconnected');
      }
    },
    onStepDetected: () => {
      trackingStore.incrementStepCount();
    },
  });

  // Config - BRIGHTER lighting for visibility
  const CONFIG = {
    camera: { height: 1.7, fov: 75, near: 0.1, far: 1000 },
    movement: { speed: 0.15, runMultiplier: 2, sensitivity: 0.002 },
    fog: { color: 0x1a1a2e, density: 0.003 },  // Much lighter fog
    lighting: { ambient: 0x606080, point: 0xffdd88 }  // Brighter ambient
  };

  // Handle file upload
  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG, JPG, JPEG)');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        setUploadedImage(img);
        setShowPreview(true);
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  }, []);

  // Extract mine data from OpenCV results - defined early to be used by processImage
  const extractMineDataFn = useCallback((lines: any, contours: any, width: number, height: number): MineData => {
    const cv = window.cv;
    const walls: WallData[] = [];
    const rooms: RoomData[] = [];

    const worldScale = 0.5 * scale;
    const offsetX = width / 2;
    const offsetY = height / 2;

    // Extract walls
    for (let i = 0; i < lines.rows; i++) {
      const x1 = (lines.data32S[i * 4] - offsetX) * worldScale;
      const z1 = (lines.data32S[i * 4 + 1] - offsetY) * worldScale;
      const x2 = (lines.data32S[i * 4 + 2] - offsetX) * worldScale;
      const z2 = (lines.data32S[i * 4 + 3] - offsetY) * worldScale;

      const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(z2 - z1, 2));
      if (length > 1) {
        walls.push({
          start: { x: x1, z: z1 },
          end: { x: x2, z: z2 },
          height: wallHeight,
          length
        });
      }
    }

    // Extract rooms
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      if (area > 500) {
        const rect = cv.boundingRect(contour);
        const roomWidth = rect.width * worldScale;
        const roomDepth = rect.height * worldScale;
        const roomCenterX = (rect.x + rect.width / 2 - offsetX) * worldScale;
        const roomCenterZ = (rect.y + rect.height / 2 - offsetY) * worldScale;

        if (roomWidth > 2 && roomDepth > 2) {
          rooms.push({
            center: { x: roomCenterX, z: roomCenterZ },
            width: roomWidth,
            depth: roomDepth,
            area: roomWidth * roomDepth
          });
        }
      }
    }

    // Calculate bounds
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    walls.forEach(wall => {
      minX = Math.min(minX, wall.start.x, wall.end.x);
      maxX = Math.max(maxX, wall.start.x, wall.end.x);
      minZ = Math.min(minZ, wall.start.z, wall.end.z);
      maxZ = Math.max(maxZ, wall.start.z, wall.end.z);
    });

    return {
      walls,
      rooms,
      bounds: { minX, maxX, minZ, maxZ },
      wallHeight,
      imageWidth: width,
      imageHeight: height
    };
  }, [scale, wallHeight]);

  // Fallback mine data
  const createFallbackMineData = useCallback((): MineData => {
    const walls: WallData[] = [
      { start: { x: -4, z: 0 }, end: { x: -4, z: -80 }, height: wallHeight, length: 80 },
      { start: { x: 4, z: 0 }, end: { x: 4, z: -80 }, height: wallHeight, length: 80 },
      { start: { x: -4, z: -80 }, end: { x: 4, z: -80 }, height: wallHeight, length: 8 },
      { start: { x: -4, z: -20 }, end: { x: -25, z: -20 }, height: wallHeight, length: 21 },
      { start: { x: -4, z: -25 }, end: { x: -25, z: -25 }, height: wallHeight, length: 21 },
      { start: { x: -25, z: -20 }, end: { x: -25, z: -25 }, height: wallHeight, length: 5 },
      { start: { x: 4, z: -40 }, end: { x: 25, z: -40 }, height: wallHeight, length: 21 },
      { start: { x: 4, z: -45 }, end: { x: 25, z: -45 }, height: wallHeight, length: 21 },
      { start: { x: 25, z: -40 }, end: { x: 25, z: -45 }, height: wallHeight, length: 5 },
    ];

    const rooms: RoomData[] = [
      { center: { x: 0, z: -40 }, width: 8, depth: 80, area: 640 },
      { center: { x: -15, z: -22.5 }, width: 21, depth: 5, area: 105 },
      { center: { x: 15, z: -42.5 }, width: 21, depth: 5, area: 105 },
    ];

    return {
      walls,
      rooms,
      bounds: { minX: -25, maxX: 25, minZ: -80, maxZ: 0 },
      wallHeight
    };
  }, [wallHeight]);

  // Process image with OpenCV
  const processImage = useCallback(() => {
    if (!cvReady || !originalCanvasRef.current || !processedCanvasRef.current || !uploadedImage) {
      console.log('processImage: not ready yet', { cvReady, hasOriginalCanvas: !!originalCanvasRef.current, hasProcessedCanvas: !!processedCanvasRef.current, hasImage: !!uploadedImage });
      return;
    }

    const cv = window.cv;
    if (!cv || !cv.Mat) {
      console.log('OpenCV not fully initialized, retrying...');
      setTimeout(processImage, 100);
      return;
    }

    try {
      const originalCanvas = originalCanvasRef.current;
      const processedCanvas = processedCanvasRef.current;

      console.log('Processing image with OpenCV...', { width: originalCanvas.width, height: originalCanvas.height });

      // Read image from canvas
      let src = cv.imread(originalCanvas);
      let dst = new cv.Mat();
      let gray = new cv.Mat();
      let edges = new cv.Mat();
      let lines = new cv.Mat();
      let contours = new cv.MatVector();
      let hierarchy = new cv.Mat();

      console.log('Source image loaded:', src.rows, 'x', src.cols);

      // Convert to grayscale
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      // Apply Gaussian blur
      cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);

      // Canny edge detection
      cv.Canny(gray, edges, threshold * 0.5, threshold, 3, false);

      // Dilate edges
      let kernel = cv.Mat.ones(3, 3, cv.CV_8U);
      cv.dilate(edges, edges, kernel);

      // Find contours
      cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      // Detect lines using Hough Transform
      cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, 30, 10);

      console.log('Detected lines:', lines.rows, 'contours:', contours.size());

      // Extract mine data
      const mineData = extractMineDataFn(lines, contours, originalCanvas.width, originalCanvas.height);
      console.log('Extracted mine data:', {
        wallCount: mineData.walls.length,
        roomCount: mineData.rooms.length,
        bounds: mineData.bounds,
        sampleWalls: mineData.walls.slice(0, 3)
      });
      setProcessedData(mineData);

      // Set up processed canvas dimensions
      processedCanvas.width = originalCanvas.width;
      processedCanvas.height = originalCanvas.height;

      // Create output image (black background)
      dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC4);

      // Draw walls (lines) in green
      for (let i = 0; i < lines.rows; i++) {
        const x1 = lines.data32S[i * 4];
        const y1 = lines.data32S[i * 4 + 1];
        const x2 = lines.data32S[i * 4 + 2];
        const y2 = lines.data32S[i * 4 + 3];
        cv.line(dst, new cv.Point(x1, y1), new cv.Point(x2, y2), [0, 255, 0, 255], 2);
      }

      // Draw contours (rooms) in orange
      for (let i = 0; i < contours.size(); i++) {
        cv.drawContours(dst, contours, i, [255, 128, 0, 255], 1, cv.LINE_8, hierarchy, 0);
      }

      // Display result on processed canvas
      cv.imshow(processedCanvas, dst);

      console.log('Image processing complete. Walls:', mineData.walls.length, 'Rooms:', mineData.rooms.length);

      // Cleanup OpenCV matrices
      src.delete();
      dst.delete();
      gray.delete();
      edges.delete();
      lines.delete();
      contours.delete();
      hierarchy.delete();
      kernel.delete();

    } catch (err) {
      console.error('OpenCV processing error:', err);
      // Fallback to demo mine data
      setProcessedData(createFallbackMineData());
    }
  }, [cvReady, threshold, uploadedImage, extractMineDataFn, createFallbackMineData]);

  // Check if scripts are already loaded on mount
  useEffect(() => {
    // Check if Three.js is already available
    if (window.THREE && !threeReady) {
      setThreeReady(true);
      console.log('Three.js already available');
    }
    // Check if OpenCV is already available
    if (window.cv && window.cv.Mat && !cvReady) {
      setCvReady(true);
      console.log('OpenCV already available');
    }
  }, [threeReady, cvReady]);

  // Draw original image on canvas
  useEffect(() => {
    if (uploadedImage && originalCanvasRef.current) {
      const canvas = originalCanvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const maxSize = 400;
      const imgScale = Math.min(maxSize / uploadedImage.width, maxSize / uploadedImage.height);
      canvas.width = uploadedImage.width * imgScale;
      canvas.height = uploadedImage.height * imgScale;
      ctx.drawImage(uploadedImage, 0, 0, canvas.width, canvas.height);

      console.log('Original image drawn to canvas:', canvas.width, 'x', canvas.height);
    }
  }, [uploadedImage]);

  // Process image when OpenCV is ready and image is loaded
  useEffect(() => {
    if (cvReady && uploadedImage && originalCanvasRef.current && showPreview) {
      // Give the canvas a moment to render
      const timer = setTimeout(() => {
        processImage();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [cvReady, uploadedImage, showPreview, processImage]);

  // Re-process when threshold or scale settings change
  useEffect(() => {
    if (showPreview && cvReady && uploadedImage && originalCanvasRef.current) {
      processImage();
    }
  }, [threshold, scale, showPreview, cvReady, uploadedImage, processImage]);

  // Update minimap
  const updateMinimap = useCallback((camera: any, mineData: MineData) => {
    const canvas = minimapCanvasRef.current;
    if (!canvas || !mineData.bounds) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { minX, maxX, minZ, maxZ } = mineData.bounds;
    const mapWidth = canvas.width;
    const mapHeight = canvas.height;
    const padding = 10;

    // Calculate scale to fit the mine in the minimap
    const mineWidth = maxX - minX;
    const mineDepth = maxZ - minZ;
    const scaleX = (mapWidth - padding * 2) / mineWidth;
    const scaleZ = (mapHeight - padding * 2) / mineDepth;
    const mapScale = Math.min(scaleX, scaleZ);

    // Helper to convert world coords to minimap coords
    const toMapX = (x: number) => padding + (x - minX) * mapScale;
    const toMapZ = (z: number) => padding + (z - minZ) * mapScale;

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(0, 0, mapWidth, mapHeight);

    // Draw walls
    ctx.strokeStyle = '#4ade80'; // Green
    ctx.lineWidth = 2;
    mineData.walls.forEach(wall => {
      ctx.beginPath();
      ctx.moveTo(toMapX(wall.start.x), toMapZ(wall.start.z));
      ctx.lineTo(toMapX(wall.end.x), toMapZ(wall.end.z));
      ctx.stroke();
    });

    // Draw rooms as filled areas
    ctx.fillStyle = 'rgba(74, 222, 128, 0.1)';
    mineData.rooms.forEach(room => {
      const rx = toMapX(room.center.x - room.width / 2);
      const rz = toMapZ(room.center.z - room.depth / 2);
      const rw = room.width * mapScale;
      const rd = room.depth * mapScale;
      ctx.fillRect(rx, rz, rw, rd);
    });

    // Draw player position
    const playerX = toMapX(camera.position.x);
    const playerZ = toMapZ(camera.position.z);

    // Player direction indicator (triangle)
    ctx.save();
    ctx.translate(playerX, playerZ);
    ctx.rotate(-eulerRef.current.y); // Rotate based on camera direction

    ctx.fillStyle = '#f97316'; // Orange
    ctx.beginPath();
    ctx.moveTo(0, -8); // Point forward
    ctx.lineTo(-5, 5);
    ctx.lineTo(5, 5);
    ctx.closePath();
    ctx.fill();

    ctx.restore();

    // Draw player dot
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(playerX, playerZ, 3, 0, Math.PI * 2);
    ctx.fill();

    // Draw border
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, mapWidth, mapHeight);
  }, []);

  // Generate 3D environment
  const generate3DEnvironment = useCallback(async () => {
    if (!processedData || !threeReady) {
      alert('Please upload and process a blueprint first');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);

    const steps = [
      { progress: 20, status: 'Creating 3D scene...' },
      { progress: 40, status: 'Building walls and floors...' },
      { progress: 60, status: 'Setting up lighting...' },
      { progress: 80, status: 'Adding particles and effects...' },
      { progress: 100, status: 'Finalizing environment...' }
    ];

    for (const step of steps) {
      setProcessingProgress(step.progress);
      setProcessingStatus(step.status);
      await new Promise(resolve => setTimeout(resolve, 400));
    }

    // Save mine data to store
    trackingStore.setMineData(processedData, 'Generated Mine');

    setIsProcessing(false);
    // Show mode selection instead of directly showing viewer
    setShowModeSelection(true);
  }, [processedData, threeReady, trackingStore]);

  // Start 3D viewer with selected mode
  const startViewer = useCallback((mode: 'track' | 'explore') => {
    trackingStore.setViewMode(mode);
    if (mode === 'track') {
      trackingStore.setIsTracking(true);
      trackingStore.resetWorkerPosition();
    } else {
      trackingStore.setIsTracking(false);
    }
    setShowModeSelection(false);
    setShowViewer(true);
    // Initialize 3D scene after state update (init3DScene defined below)
  }, [trackingStore]);

  // ==================== SIMULATION CONTROL ====================
  const startSimulation = useCallback(() => {
    if (simulationRef.current) return;

    setIsSimulating(true);
    let simHeading = 0;
    let stepCount = 0;

    simulationRef.current = setInterval(() => {
      const state = useMineTrackingStore.getState();
      const bounds = state.mineData?.bounds;

      // Random heading changes for realistic movement
      simHeading += (Math.random() - 0.5) * 0.3;

      // Calculate new position
      const stepLength = 1.5;
      let newX = state.workerPosition.x + stepLength * Math.sin(simHeading);
      let newZ = state.workerPosition.z - stepLength * Math.cos(simHeading);

      // Keep within bounds if available
      if (bounds) {
        const margin = 3;
        if (newX < bounds.minX + margin || newX > bounds.maxX - margin) {
          simHeading = Math.PI - simHeading; // Reverse X direction
          newX = state.workerPosition.x + stepLength * Math.sin(simHeading);
        }
        if (newZ < bounds.minZ + margin || newZ > bounds.maxZ - margin) {
          simHeading = -simHeading; // Reverse Z direction
          newZ = state.workerPosition.z - stepLength * Math.cos(simHeading);
        }
        newX = Math.max(bounds.minX + margin, Math.min(bounds.maxX - margin, newX));
        newZ = Math.max(bounds.minZ + margin, Math.min(bounds.maxZ - margin, newZ));
      }

      stepCount++;
      console.log(`[SIM] Step #${stepCount} pos=(${newX.toFixed(1)}, ${newZ.toFixed(1)}) heading=${(simHeading * 180 / Math.PI).toFixed(0)}°`);

      // Update store
      state.updateWorkerPosition({ x: newX, y: 1.7, z: newZ }, simHeading);
      state.incrementStepCount();
    }, 500); // Step every 500ms
  }, []);

  const stopSimulation = useCallback(() => {
    if (simulationRef.current) {
      clearInterval(simulationRef.current);
      simulationRef.current = null;
    }
    setIsSimulating(false);
  }, []);

  // Cleanup simulation on unmount
  useEffect(() => {
    return () => {
      if (simulationRef.current) {
        clearInterval(simulationRef.current);
      }
    };
  }, []);

  // ==================== 2D MAP RENDERING ====================
  const render2DMap = useCallback(() => {
    const canvas = map2DCanvasRef.current;
    const mineData = trackingStore.mineData;
    if (!canvas || !mineData?.bounds) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { minX, maxX, minZ, maxZ } = mineData.bounds;
    const mapWidth = canvas.width;
    const mapHeight = canvas.height;
    const padding = 40;

    // Calculate scale to fit mine in canvas
    const mineWidth = maxX - minX;
    const mineDepth = maxZ - minZ;
    const scaleX = (mapWidth - padding * 2) / mineWidth;
    const scaleZ = (mapHeight - padding * 2) / mineDepth;
    const mapScale = Math.min(scaleX, scaleZ) * 0.85;

    // Helper to convert world coords to canvas coords
    const toMapX = (x: number) => padding + (x - minX + 2) * mapScale;
    const toMapZ = (z: number) => padding + (z - minZ + 2) * mapScale;

    // Clear canvas with dark background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, mapWidth, mapHeight);

    // Draw grid
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.2)';
    ctx.lineWidth = 1;
    for (let x = Math.floor(minX); x <= Math.ceil(maxX); x += 5) {
      ctx.beginPath();
      ctx.moveTo(toMapX(x), toMapZ(minZ - 2));
      ctx.lineTo(toMapX(x), toMapZ(maxZ + 2));
      ctx.stroke();
    }
    for (let z = Math.floor(minZ); z <= Math.ceil(maxZ); z += 5) {
      ctx.beginPath();
      ctx.moveTo(toMapX(minX - 2), toMapZ(z));
      ctx.lineTo(toMapX(maxX + 2), toMapZ(z));
      ctx.stroke();
    }

    // Draw rooms as filled areas
    ctx.fillStyle = 'rgba(74, 222, 128, 0.15)';
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.3)';
    ctx.lineWidth = 1;
    mineData.rooms.forEach(room => {
      const rx = toMapX(room.center.x - room.width / 2);
      const rz = toMapZ(room.center.z - room.depth / 2);
      const rw = room.width * mapScale;
      const rd = room.depth * mapScale;
      ctx.fillRect(rx, rz, rw, rd);
      ctx.strokeRect(rx, rz, rw, rd);
    });

    // Draw walls with glow effect
    ctx.shadowColor = '#4ade80';
    ctx.shadowBlur = 5;
    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth = 3;
    mineData.walls.forEach(wall => {
      ctx.beginPath();
      ctx.moveTo(toMapX(wall.start.x), toMapZ(wall.start.z));
      ctx.lineTo(toMapX(wall.end.x), toMapZ(wall.end.z));
      ctx.stroke();
    });
    ctx.shadowBlur = 0;

    // Draw entrance marker
    const entrance = useMineTrackingStore.getState().entrancePosition;
    const entranceX = toMapX(entrance.x);
    const entranceZ = toMapZ(entrance.z);

    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.arc(entranceX, entranceZ, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('GATE', entranceX, entranceZ + 25);

    // Draw worker position with trail effect
    const worker = useMineTrackingStore.getState().workerPosition;
    const heading = useMineTrackingStore.getState().workerHeading;
    const workerX = toMapX(worker.x);
    const workerZ = toMapZ(worker.z);

    // Pulsing outer glow
    const pulse = Math.sin(Date.now() / 150) * 0.4 + 0.6;
    ctx.fillStyle = `rgba(249, 115, 22, ${0.3 * pulse})`;
    ctx.beginPath();
    ctx.arc(workerX, workerZ, 25, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = `rgba(249, 115, 22, ${0.5 * pulse})`;
    ctx.beginPath();
    ctx.arc(workerX, workerZ, 18, 0, Math.PI * 2);
    ctx.fill();

    // Direction indicator (triangle)
    ctx.save();
    ctx.translate(workerX, workerZ);
    ctx.rotate(-heading + Math.PI);

    ctx.fillStyle = '#f97316';
    ctx.beginPath();
    ctx.moveTo(0, -18);
    ctx.lineTo(-10, 10);
    ctx.lineTo(10, 10);
    ctx.closePath();
    ctx.fill();

    // Worker center
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(0, 0, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Draw coordinates and info
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Worker: (${worker.x.toFixed(1)}, ${worker.z.toFixed(1)})`, 10, mapHeight - 30);
    ctx.fillText(`Heading: ${(heading * 180 / Math.PI).toFixed(0)}°`, 10, mapHeight - 15);

    ctx.textAlign = 'right';
    ctx.fillText(`Steps: ${useMineTrackingStore.getState().stepCount}`, mapWidth - 10, mapHeight - 15);
  }, [trackingStore.mineData]);

  // 2D Map animation loop
  useEffect(() => {
    if (!show2DMap) return;

    let animFrame: number;
    const animate2D = () => {
      render2DMap();
      animFrame = requestAnimationFrame(animate2D);
    };
    animate2D();

    return () => cancelAnimationFrame(animFrame);
  }, [show2DMap, render2DMap]);

  // Effect to initialize scene when viewer is shown
  useEffect(() => {
    if (showViewer && processedData && threeReady) {
      // Small delay to ensure DOM is ready
      const timer = setTimeout(() => {
        if (viewerContainerRef.current) {
          init3DScene();
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [showViewer, processedData, threeReady]);

  // Initialize Three.js scene
  const init3DScene = useCallback(() => {
    if (!viewerContainerRef.current || !processedData) return;

    const THREE = window.THREE;
    const mineData = processedData;

    console.log('Initializing 3D scene with mine data:', {
      wallCount: mineData.walls.length,
      roomCount: mineData.rooms.length,
      bounds: mineData.bounds
    });

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.fog.color);
    scene.fog = new THREE.FogExp2(CONFIG.fog.color, CONFIG.fog.density);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      CONFIG.camera.fov,
      viewerContainerRef.current.clientWidth / viewerContainerRef.current.clientHeight,
      CONFIG.camera.near,
      CONFIG.camera.far
    );
    const startZ = mineData.bounds ? (mineData.bounds.maxZ + mineData.bounds.minZ) / 2 : 0;
    camera.position.set(0, CONFIG.camera.height, startZ + 5);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(viewerContainerRef.current.clientWidth, viewerContainerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    viewerContainerRef.current.innerHTML = '';
    viewerContainerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    clockRef.current = new THREE.Clock();

    // Build the mine
    createLighting(scene, mineData);
    createFlashlight(scene, camera);
    createMineFromData(scene, mineData);
    if (addDust) createDustParticles(scene, mineData);

    // Create entrance marker
    const entrancePos = trackingStore.entrancePosition;
    const entranceMarker = createEntranceMarker(entrancePos);
    scene.add(entranceMarker);
    entranceMarkerRef.current = entranceMarker;

    // Create worker avatar (only visible in track mode or when tracking is active)
    const workerAvatar = createWorkerAvatar();
    workerAvatar.position.set(entrancePos.x, 0, entrancePos.z);
    scene.add(workerAvatar);
    workerMeshRef.current = workerAvatar;

    // Set initial camera position based on mode
    if (trackingStore.viewMode === 'track') {
      // Position camera behind worker at entrance
      camera.position.set(
        entrancePos.x + CAMERA_FOLLOW_OFFSET.x,
        CAMERA_FOLLOW_OFFSET.y,
        entrancePos.z + CAMERA_FOLLOW_OFFSET.z
      );
      camera.lookAt(entrancePos.x, 1.7, entrancePos.z);
    }

    // Setup controls (only active in explore mode)
    setupControls(renderer.domElement, camera);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      const delta = clockRef.current.getDelta();

      // Get FRESH state from store on each frame (critical for real-time updates!)
      const currentState = useMineTrackingStore.getState();
      const currentViewMode = currentState.viewMode;
      const workerPos = currentState.workerPosition;
      const workerHeading = currentState.workerHeading;
      const stepCount = currentState.stepCount;

      // Debug log every ~60 frames (1 second at 60fps)
      if (Math.random() < 0.016) {
        console.log(`[3D] Worker pos=(${workerPos.x.toFixed(2)}, ${workerPos.z.toFixed(2)}) heading=${(workerHeading * 180 / Math.PI).toFixed(1)}° steps=${stepCount}`);
      }

      // Update worker avatar position and rotation
      if (workerMeshRef.current) {
        workerMeshRef.current.position.set(workerPos.x, 0, workerPos.z);
        workerMeshRef.current.rotation.y = -workerHeading; // Negative because Three.js Y rotation is opposite
      }

      if (currentViewMode === 'track') {
        // Track mode: Camera follows worker
        // Calculate target camera position (behind and above worker based on heading)
        const offsetX = Math.sin(workerHeading) * CAMERA_FOLLOW_OFFSET.z;
        const offsetZ = Math.cos(workerHeading) * CAMERA_FOLLOW_OFFSET.z;

        targetCameraPositionRef.current = {
          x: workerPos.x + offsetX,
          y: CAMERA_FOLLOW_OFFSET.y,
          z: workerPos.z + offsetZ
        };
        targetCameraLookAtRef.current = {
          x: workerPos.x,
          y: 1.7,
          z: workerPos.z
        };

        // Smooth camera follow using lerp
        camera.position.x += (targetCameraPositionRef.current.x - camera.position.x) * CAMERA_LERP_FACTOR;
        camera.position.y += (targetCameraPositionRef.current.y - camera.position.y) * CAMERA_LERP_FACTOR;
        camera.position.z += (targetCameraPositionRef.current.z - camera.position.z) * CAMERA_LERP_FACTOR;

        // Make camera look at worker
        camera.lookAt(
          targetCameraLookAtRef.current.x,
          targetCameraLookAtRef.current.y,
          targetCameraLookAtRef.current.z
        );
      } else {
        // Explore mode: Manual camera control
        const moveState = moveStateRef.current;
        if (moveState.forward || moveState.backward || moveState.left || moveState.right) {
          const speed = CONFIG.movement.speed * (moveState.running ? CONFIG.movement.runMultiplier : 1);

          const direction = new THREE.Vector3();
          direction.z = Number(moveState.forward) - Number(moveState.backward);
          direction.x = Number(moveState.right) - Number(moveState.left);
          direction.normalize();

          const velocity = new THREE.Vector3(direction.x * speed, 0, direction.z * speed);
          velocity.applyQuaternion(camera.quaternion);
          velocity.y = 0;
          camera.position.add(velocity);
        }

        // Keep camera at proper height in explore mode
        camera.position.y = CONFIG.camera.height;
      }

      // Bound camera
      if (mineData.bounds) {
        const { minX, maxX, minZ, maxZ } = mineData.bounds;
        camera.position.x = Math.max(minX - 10, Math.min(maxX + 10, camera.position.x));
        camera.position.z = Math.max(minZ - 10, Math.min(maxZ + 10, camera.position.z));
      }

      // Update minimap with worker position
      updateMinimapWithWorker(camera, mineData, workerPos, workerHeading);

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!viewerContainerRef.current) return;
      camera.aspect = viewerContainerRef.current.clientWidth / viewerContainerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(viewerContainerRef.current.clientWidth, viewerContainerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
    // Note: updateMinimapWithWorker is defined below but stable due to useCallback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [processedData, addDust, addLights, trackingStore]);

  // Create lighting
  const createLighting = (scene: any, mineData: MineData) => {
    const THREE = window.THREE;

    // Strong ambient light for visibility
    const ambient = new THREE.AmbientLight(CONFIG.lighting.ambient, 0.8);
    scene.add(ambient);

    // Add directional light from above for overall illumination
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(0, 50, 0);
    scene.add(dirLight);

    if (!addLights) return;

    const lightPositions: { x: number; y: number; z: number }[] = [];

    if (mineData.rooms.length > 0) {
      mineData.rooms.forEach(room => {
        lightPositions.push({ x: room.center.x, y: 3, z: room.center.z });
      });
    }

    if (mineData.bounds) {
      const { minX, maxX, minZ, maxZ } = mineData.bounds;
      const midX = (minX + maxX) / 2;
      const midZ = (minZ + maxZ) / 2;
      lightPositions.push({ x: midX, y: 3, z: minZ + 5 });
      lightPositions.push({ x: midX, y: 3, z: maxZ - 5 });
      lightPositions.push({ x: midX, y: 3, z: midZ });
    }

    if (lightPositions.length === 0) {
      lightPositions.push({ x: 0, y: 3, z: -15 }, { x: 0, y: 3, z: -30 }, { x: 0, y: 3, z: -45 });
    }

    lightPositions.forEach(pos => {
      const light = new THREE.PointLight(CONFIG.lighting.point, 1, 30);
      light.position.set(pos.x, pos.y, pos.z);
      light.castShadow = true;
      scene.add(light);

      const fixtureGeom = new THREE.CylinderGeometry(0.1, 0.2, 0.3, 8);
      const fixtureMat = new THREE.MeshBasicMaterial({ color: 0xffaa44 });
      const fixture = new THREE.Mesh(fixtureGeom, fixtureMat);
      fixture.position.set(pos.x, pos.y + 0.3, pos.z);
      scene.add(fixture);
    });
  };

  // Create flashlight
  const createFlashlight = (scene: any, camera: any) => {
    const THREE = window.THREE;

    const flashlight = new THREE.SpotLight(0xffffff, 2, 50, Math.PI / 6, 0.3, 1);
    flashlight.castShadow = true;
    flashlightRef.current = flashlight;

    const flashlightTarget = new THREE.Object3D();
    flashlightTarget.position.set(0, 0, -1);
    scene.add(flashlightTarget);
    flashlight.target = flashlightTarget;

    camera.add(flashlight);
    flashlight.position.set(0.3, -0.2, 0);
    scene.add(camera);

    const playerLight = new THREE.PointLight(0xffffee, 0.5, 8);
    camera.add(playerLight);
  };

  // Create entrance marker (gate) - Coal mine worker statue in orange
  const createEntranceMarker = (position: { x: number; z: number }) => {
    const THREE = window.THREE;

    // Create a group to hold all entrance marker elements
    const group = new THREE.Group();

    // === COAL MINE WORKER STATUE ===

    // Boots (dark gray)
    const bootMat = new THREE.MeshStandardMaterial({ color: 0x1f2937 });
    const leftBoot = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.3, 0.35), bootMat);
    leftBoot.position.set(-0.15, 0.15, 0);
    group.add(leftBoot);
    const rightBoot = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.3, 0.35), bootMat);
    rightBoot.position.set(0.15, 0.15, 0);
    group.add(rightBoot);

    // Legs (dark pants)
    const pantsMat = new THREE.MeshStandardMaterial({ color: 0x374151 });
    const leftLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.8, 8), pantsMat);
    leftLeg.position.set(-0.15, 0.7, 0);
    group.add(leftLeg);
    const rightLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.8, 8), pantsMat);
    rightLeg.position.set(0.15, 0.7, 0);
    group.add(rightLeg);

    // Torso (orange high-vis vest)
    const vestMat = new THREE.MeshStandardMaterial({
      color: 0xf97316, // Orange
      emissive: 0xf97316,
      emissiveIntensity: 0.15,
    });
    const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.2, 0.9, 8), vestMat);
    torso.position.y = 1.55;
    group.add(torso);

    // Reflective stripes on vest (bright yellow)
    const stripeMat = new THREE.MeshBasicMaterial({ color: 0xfde047 });
    const stripe1 = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.06, 0.01), stripeMat);
    stripe1.position.set(0, 1.7, 0.22);
    group.add(stripe1);
    const stripe2 = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.06, 0.01), stripeMat);
    stripe2.position.set(0, 1.4, 0.22);
    group.add(stripe2);

    // Arms (orange sleeves)
    const leftArm = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.6, 8), vestMat);
    leftArm.position.set(-0.35, 1.5, 0);
    leftArm.rotation.z = 0.3;
    group.add(leftArm);
    const rightArm = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.6, 8), vestMat);
    rightArm.position.set(0.35, 1.5, 0);
    rightArm.rotation.z = -0.3;
    group.add(rightArm);

    // Skin material for neck and head
    const skinMat = new THREE.MeshStandardMaterial({ color: 0xd4a574 });

    // Neck
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.15, 8), skinMat);
    neck.position.y = 2.07;
    group.add(neck);

    // Head (face)
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 16), skinMat);
    head.position.y = 2.25;
    group.add(head);

    // Hard hat (yellow safety helmet)
    const helmetMat = new THREE.MeshStandardMaterial({
      color: 0xfbbf24, // Yellow
      emissive: 0xfbbf24,
      emissiveIntensity: 0.2,
    });
    const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), helmetMat);
    helmet.position.y = 2.32;
    group.add(helmet);

    // Helmet brim
    const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.03, 16), helmetMat);
    brim.position.y = 2.2;
    group.add(brim);

    // Headlamp on helmet
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.03, 8), lampMat);
    lamp.position.set(0, 2.28, 0.2);
    lamp.rotation.x = Math.PI / 2;
    group.add(lamp);

    // === PEDESTAL/BASE ===
    const pedestalMat = new THREE.MeshStandardMaterial({ color: 0x4b5563 });
    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.7, 0.2, 16), pedestalMat);
    pedestal.position.y = -0.1;
    group.add(pedestal);

    // "GATE" sign on pedestal
    const signMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    const signBack = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.25, 0.05), signMat);
    signBack.position.set(0, -0.05, 0.65);
    group.add(signBack);

    // Glowing ring around base
    const ringGeom = new THREE.TorusGeometry(0.8, 0.05, 8, 32);
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      emissive: 0x22c55e,
      emissiveIntensity: 0.4,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.02;
    group.add(ring);

    // Point light for visibility
    const light = new THREE.PointLight(0xfbbf24, 0.8, 8);
    light.position.y = 2.5;
    group.add(light);

    group.position.set(position.x, 0, position.z);
    return group;
  };

  // Create worker avatar
  const createWorkerAvatar = () => {
    const THREE = window.THREE;

    // Create a group for the worker
    const group = new THREE.Group();

    // Body (capsule-like shape using cylinder + spheres)
    const bodyGeom = new THREE.CylinderGeometry(0.25, 0.25, 1, 16);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xf97316, // Orange
      emissive: 0xf97316,
      emissiveIntensity: 0.2,
    });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    body.position.y = 1;
    group.add(body);

    // Head (sphere)
    const headGeom = new THREE.SphereGeometry(0.2, 16, 16);
    const headMat = new THREE.MeshStandardMaterial({
      color: 0xfbbf24, // Yellow/amber for helmet
      emissive: 0xfbbf24,
      emissiveIntensity: 0.3,
    });
    const head = new THREE.Mesh(headGeom, headMat);
    head.position.y = 1.7;
    group.add(head);

    // Direction indicator (cone pointing forward)
    const arrowGeom = new THREE.ConeGeometry(0.15, 0.4, 8);
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0xef4444 }); // Red
    const arrow = new THREE.Mesh(arrowGeom, arrowMat);
    arrow.rotation.x = Math.PI / 2;
    arrow.position.set(0, 1.2, -0.4);
    group.add(arrow);

    // Light around worker for visibility
    const workerLight = new THREE.PointLight(0xf97316, 0.5, 5);
    workerLight.position.y = 1.5;
    group.add(workerLight);

    return group;
  };

  // Enhanced minimap with worker tracking
  const updateMinimapWithWorker = useCallback((
    camera: any,
    mineData: MineData,
    workerPos: { x: number; y: number; z: number },
    workerHeading: number
  ) => {
    const canvas = minimapCanvasRef.current;
    if (!canvas || !mineData.bounds) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { minX, maxX, minZ, maxZ } = mineData.bounds;
    const mapWidth = canvas.width;
    const mapHeight = canvas.height;
    const padding = 10;

    // Calculate scale to fit the mine in the minimap
    const mineWidth = maxX - minX;
    const mineDepth = maxZ - minZ;
    const scaleX = (mapWidth - padding * 2) / mineWidth;
    const scaleZ = (mapHeight - padding * 2) / mineDepth;
    const mapScale = Math.min(scaleX, scaleZ) * 0.9; // Slightly smaller to fit entrance

    // Helper to convert world coords to minimap coords
    const toMapX = (x: number) => padding + (x - minX + 2) * mapScale;
    const toMapZ = (z: number) => padding + (z - minZ + 2) * mapScale;

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fillRect(0, 0, mapWidth, mapHeight);

    // Draw walls
    ctx.strokeStyle = '#4ade80'; // Green
    ctx.lineWidth = 2;
    mineData.walls.forEach(wall => {
      ctx.beginPath();
      ctx.moveTo(toMapX(wall.start.x), toMapZ(wall.start.z));
      ctx.lineTo(toMapX(wall.end.x), toMapZ(wall.end.z));
      ctx.stroke();
    });

    // Draw rooms as filled areas
    ctx.fillStyle = 'rgba(74, 222, 128, 0.1)';
    mineData.rooms.forEach(room => {
      const rx = toMapX(room.center.x - room.width / 2);
      const rz = toMapZ(room.center.z - room.depth / 2);
      const rw = room.width * mapScale;
      const rd = room.depth * mapScale;
      ctx.fillRect(rx, rz, rw, rd);
    });

    // Draw entrance marker (get fresh state)
    const currentEntrance = useMineTrackingStore.getState().entrancePosition;
    const entranceX = toMapX(currentEntrance.x);
    const entranceZ = toMapZ(currentEntrance.z);
    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.arc(entranceX, entranceZ, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 8px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('GATE', entranceX, entranceZ + 15);

    // Draw worker position (pulsing orange dot)
    const workerX = toMapX(workerPos.x);
    const workerZ = toMapZ(workerPos.z);

    // Pulsing effect
    const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7;

    // Outer glow
    ctx.fillStyle = `rgba(249, 115, 22, ${0.3 * pulse})`;
    ctx.beginPath();
    ctx.arc(workerX, workerZ, 10, 0, Math.PI * 2);
    ctx.fill();

    // Worker direction indicator (triangle)
    ctx.save();
    ctx.translate(workerX, workerZ);
    ctx.rotate(-workerHeading + Math.PI); // Adjust rotation for map orientation

    ctx.fillStyle = '#f97316'; // Orange
    ctx.beginPath();
    ctx.moveTo(0, -10); // Point forward
    ctx.lineTo(-6, 6);
    ctx.lineTo(6, 6);
    ctx.closePath();
    ctx.fill();

    ctx.restore();

    // Worker center dot
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(workerX, workerZ, 3, 0, Math.PI * 2);
    ctx.fill();

    // Draw camera position (only in explore mode)
    if (trackingStore.viewMode === 'explore') {
      const camX = toMapX(camera.position.x);
      const camZ = toMapZ(camera.position.z);

      ctx.save();
      ctx.translate(camX, camZ);
      ctx.rotate(-eulerRef.current.y);

      ctx.fillStyle = '#3b82f6'; // Blue
      ctx.beginPath();
      ctx.moveTo(0, -8);
      ctx.lineTo(-5, 5);
      ctx.lineTo(5, 5);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }

    // Draw border
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, mapWidth, mapHeight);

    // Connection status indicator
    const statusColor = trackingStore.connectionStatus === 'connected' ? '#22c55e' :
                        trackingStore.connectionStatus === 'simulating' ? '#fbbf24' : '#ef4444';
    ctx.fillStyle = statusColor;
    ctx.beginPath();
    ctx.arc(mapWidth - 12, 12, 5, 0, Math.PI * 2);
    ctx.fill();
  }, [trackingStore]);

  // Create mine geometry
  const createMineFromData = (scene: any, mineData: MineData) => {
    const THREE = window.THREE;

    // Create textures
    const rockTexture = createRockTexture();
    const floorTexture = createFloorTexture();

    const wallMaterial = new THREE.MeshStandardMaterial({
      map: rockTexture,
      roughness: 0.9,
      metalness: 0.1,
      bumpMap: rockTexture,
      bumpScale: 0.3,
      side: THREE.DoubleSide  // Make walls visible from both sides
    });

    const floorMaterial = new THREE.MeshStandardMaterial({
      map: floorTexture,
      roughness: 0.95,
      metalness: 0.05
    });

    const ceilingMaterial = new THREE.MeshStandardMaterial({
      color: 0x1a1a1a,
      roughness: 1,
      metalness: 0
    });

    // Floor and ceiling
    if (mineData.bounds) {
      const { minX, maxX, minZ, maxZ } = mineData.bounds;
      const width = maxX - minX + 20;
      const depth = maxZ - minZ + 20;
      const centerX = (minX + maxX) / 2;
      const centerZ = (minZ + maxZ) / 2;

      const floorGeom = new THREE.PlaneGeometry(width, depth);
      const floor = new THREE.Mesh(floorGeom, floorMaterial);
      floor.rotation.x = -Math.PI / 2;
      floor.position.set(centerX, 0, centerZ);
      floor.receiveShadow = true;
      scene.add(floor);

      const ceiling = new THREE.Mesh(floorGeom, ceilingMaterial);
      ceiling.rotation.x = Math.PI / 2;
      ceiling.position.set(centerX, mineData.wallHeight, centerZ);
      scene.add(ceiling);
    }

    // Walls - using PlaneGeometry like ar-min repo for proper wall rendering
    mineData.walls.forEach(wall => {
      const dx = wall.end.x - wall.start.x;
      const dz = wall.end.z - wall.start.z;
      const length = Math.sqrt(dx * dx + dz * dz);

      // Skip walls that are too short
      if (length < 0.5) return;

      // Calculate angle - atan2(dz, dx) gives angle from X-axis
      const angle = Math.atan2(dz, dx);

      // Create wall as a plane (like ar-min) - DoubleSide material makes it visible from both sides
      const wallGeom = new THREE.PlaneGeometry(length, wall.height);
      const wallMesh = new THREE.Mesh(wallGeom, wallMaterial);

      const centerX = (wall.start.x + wall.end.x) / 2;
      const centerZ = (wall.start.z + wall.end.z) / 2;

      wallMesh.position.set(centerX, wall.height / 2, centerZ);
      // Rotate the plane to align with the wall direction
      // PlaneGeometry faces +Z by default, so we rotate to align with the wall vector
      wallMesh.rotation.y = -angle;
      wallMesh.castShadow = true;
      wallMesh.receiveShadow = true;
      scene.add(wallMesh);
    });

    // Also log wall count for debugging
    console.log('Created', mineData.walls.length, 'walls in 3D scene');

    // Rails
    if (mineData.bounds) {
      const railMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.9, roughness: 0.3 });
      const { minZ, maxZ } = mineData.bounds;
      const railLength = maxZ - minZ;

      const railGeom = new THREE.BoxGeometry(0.1, 0.1, railLength);
      const leftRail = new THREE.Mesh(railGeom, railMat);
      leftRail.position.set(-1.5, 0.05, (minZ + maxZ) / 2);
      scene.add(leftRail);

      const rightRail = new THREE.Mesh(railGeom, railMat);
      rightRail.position.set(-0.5, 0.05, (minZ + maxZ) / 2);
      scene.add(rightRail);
    }
  };

  // Create rock texture
  const createRockTexture = () => {
    const THREE = window.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(0, 0, 256, 256);

    for (let i = 0; i < 5000; i++) {
      const x = Math.random() * 256;
      const y = Math.random() * 256;
      const gray = Math.floor(30 + Math.random() * 30);
      ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
      ctx.fillRect(x, y, 2, 2);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(4, 4);
    return texture;
  };

  // Create floor texture
  const createFloorTexture = () => {
    const THREE = window.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = '#3d3428';
    ctx.fillRect(0, 0, 256, 256);

    for (let i = 0; i < 300; i++) {
      const x = Math.random() * 256;
      const y = Math.random() * 256;
      const size = 1 + Math.random() * 4;
      const gray = Math.floor(40 + Math.random() * 30);
      ctx.fillStyle = `rgb(${gray}, ${gray - 5}, ${gray - 10})`;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(8, 8);
    return texture;
  };

  // Create dust particles
  const createDustParticles = (scene: any, mineData: MineData) => {
    const THREE = window.THREE;
    if (!mineData.bounds) return;

    const { minX, maxX, minZ, maxZ } = mineData.bounds;
    const particleCount = 500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = minX + Math.random() * (maxX - minX);
      positions[i + 1] = Math.random() * mineData.wallHeight;
      positions[i + 2] = minZ + Math.random() * (maxZ - minZ);
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0x888888,
      size: 0.05,
      transparent: true,
      opacity: 0.6
    });

    const dustParticles = new THREE.Points(geometry, material);
    scene.add(dustParticles);
  };

  // Setup controls
  const setupControls = (canvas: HTMLCanvasElement, camera: any) => {
    const THREE = window.THREE;

    canvas.addEventListener('click', () => {
      canvas.requestPointerLock();
    });

    document.addEventListener('pointerlockchange', () => {
      isPointerLockedRef.current = document.pointerLockElement === canvas;
    });

    document.addEventListener('mousemove', (event) => {
      if (!isPointerLockedRef.current) return;

      const movementX = event.movementX || 0;
      const movementY = event.movementY || 0;

      eulerRef.current.y -= movementX * CONFIG.movement.sensitivity;
      eulerRef.current.x -= movementY * CONFIG.movement.sensitivity;
      eulerRef.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, eulerRef.current.x));

      const euler = new THREE.Euler(eulerRef.current.x, eulerRef.current.y, 0, 'YXZ');
      camera.quaternion.setFromEuler(euler);
    });

    const handleKeyDown = (event: KeyboardEvent) => {
      switch (event.code) {
        case 'KeyW':
        case 'ArrowUp':
          moveStateRef.current.forward = true;
          break;
        case 'KeyS':
        case 'ArrowDown':
          moveStateRef.current.backward = true;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          moveStateRef.current.left = true;
          break;
        case 'KeyD':
        case 'ArrowRight':
          moveStateRef.current.right = true;
          break;
        case 'ShiftLeft':
        case 'ShiftRight':
          moveStateRef.current.running = true;
          break;
        case 'KeyF':
          isFlashlightOnRef.current = !isFlashlightOnRef.current;
          if (flashlightRef.current) {
            flashlightRef.current.intensity = isFlashlightOnRef.current ? 2 : 0;
          }
          break;
        case 'Escape':
          document.exitPointerLock();
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      switch (event.code) {
        case 'KeyW':
        case 'ArrowUp':
          moveStateRef.current.forward = false;
          break;
        case 'KeyS':
        case 'ArrowDown':
          moveStateRef.current.backward = false;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          moveStateRef.current.left = false;
          break;
        case 'KeyD':
        case 'ArrowRight':
          moveStateRef.current.right = false;
          break;
        case 'ShiftLeft':
        case 'ShiftRight':
          moveStateRef.current.running = false;
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
  };

  // Export mine data
  const exportMineData = () => {
    if (!processedData) return;

    const exportData = {
      name: 'Generated Mine',
      version: '1.0',
      generatedAt: new Date().toISOString(),
      settings: { wallHeight, scale },
      walls: processedData.walls,
      rooms: processedData.rooms,
      bounds: processedData.bounds
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mine_layout_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Back to upload
  const backToUpload = () => {
    setShowViewer(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (rendererRef.current) {
      rendererRef.current.dispose();
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    dropZoneRef.current?.classList.add('border-orange-500', 'bg-orange-50');
  };

  const handleDragLeave = () => {
    dropZoneRef.current?.classList.remove('border-orange-500', 'bg-orange-50');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dropZoneRef.current?.classList.remove('border-orange-500', 'bg-orange-50');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  // Calculate stats
  const wallCount = processedData?.walls.length || 0;
  const roomCount = processedData?.rooms.length || 0;
  const totalArea = processedData?.rooms.reduce((sum, room) => sum + room.area, 0).toFixed(1) || '0';

  return (
    <AppLayout>
      {/* Load Three.js */}
      <Script
        src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
        onLoad={() => {
          console.log('Three.js script loaded');
          // Check if THREE is available
          const checkThreeReady = () => {
            if (window.THREE) {
              setThreeReady(true);
              console.log('Three.js is ready');
            } else {
              setTimeout(checkThreeReady, 100);
            }
          };
          checkThreeReady();
        }}
        strategy="afterInteractive"
      />
      {/* Load OpenCV.js */}
      <Script
        src="https://docs.opencv.org/4.x/opencv.js"
        onLoad={() => {
          console.log('OpenCV.js script loaded');
          // OpenCV.js loads asynchronously and needs time to initialize
          const checkCvReady = () => {
            if (window.cv && window.cv.Mat) {
              setCvReady(true);
              console.log('OpenCV.js is ready');
            } else {
              setTimeout(checkCvReady, 100);
            }
          };
          checkCvReady();
        }}
        strategy="afterInteractive"
      />

      {/* Processing Overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-white/95 z-50 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-stone-200 border-t-orange-500 rounded-full animate-spin mx-auto mb-6" />
            <h2 className="text-xl font-semibold text-orange-600 mb-2">Processing Blueprint...</h2>
            <p className="text-stone-500 mb-4">{processingStatus}</p>
            <div className="w-64 h-2 bg-stone-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all duration-300"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Mode Selection Screen */}
      {showModeSelection && (
        <div className="fixed inset-0 bg-gradient-to-br from-stone-900 via-stone-800 to-stone-900 z-50 flex items-center justify-center">
          <div className="text-center max-w-2xl mx-auto px-6">
            {/* Header */}
            <div className="mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-orange-400 to-amber-400 mb-6">
                <Map className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-white mb-3">3D Mine Ready</h1>
              <p className="text-stone-400">Choose how you want to explore the generated mine environment</p>
            </div>

            {/* Mode Selection Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {/* Track Worker Mode */}
              <button
                onClick={() => startViewer('track')}
                className="group bg-white/10 backdrop-blur border border-white/20 rounded-2xl p-6 text-left hover:bg-orange-500/20 hover:border-orange-400 transition-all duration-300"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-xl bg-orange-500/20 flex items-center justify-center group-hover:bg-orange-500/30 transition-colors">
                    <User className="w-7 h-7 text-orange-400" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white">Track Worker</h3>
                    <p className="text-sm text-orange-400">Real-time tracking mode</p>
                  </div>
                </div>
                <p className="text-stone-400 text-sm mb-4">
                  Camera automatically follows the worker as they move through the mine. Position is tracked using helmet IMU sensors.
                </p>
                <div className="flex items-center gap-2 text-orange-400 text-sm">
                  <Radio className="w-4 h-4" />
                  <span>Live helmet connection</span>
                </div>
              </button>

              {/* Explore Mode */}
              <button
                onClick={() => startViewer('explore')}
                className="group bg-white/10 backdrop-blur border border-white/20 rounded-2xl p-6 text-left hover:bg-blue-500/20 hover:border-blue-400 transition-all duration-300"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-xl bg-blue-500/20 flex items-center justify-center group-hover:bg-blue-500/30 transition-colors">
                    <Eye className="w-7 h-7 text-blue-400" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white">Explore Mine</h3>
                    <p className="text-sm text-blue-400">Free camera mode</p>
                  </div>
                </div>
                <p className="text-stone-400 text-sm mb-4">
                  Freely explore the 3D mine environment using keyboard and mouse controls. Great for inspecting the layout.
                </p>
                <div className="flex items-center gap-2 text-blue-400 text-sm">
                  <Navigation className="w-4 h-4" />
                  <span>WASD + Mouse controls</span>
                </div>
              </button>
            </div>

            {/* Back Button */}
            <button
              onClick={() => {
                setShowModeSelection(false);
                setShowPreview(true);
              }}
              className="text-stone-400 hover:text-white transition-colors text-sm flex items-center gap-2 mx-auto"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Blueprint Settings
            </button>
          </div>
        </div>
      )}

      {/* 3D Viewer */}
      {showViewer ? (
        <div className="fixed inset-0 bg-stone-900 z-40">
          <div ref={viewerContainerRef} className="w-full h-full" />

          {/* HUD */}
          <div className="fixed inset-0 pointer-events-none z-50">
            {/* Info Panel */}
            <div className="absolute top-4 left-4 bg-white rounded-xl p-4 shadow-lg pointer-events-auto">
              <h3 className="text-xs font-semibold text-orange-600 tracking-wider mb-3 pb-2 border-b">
                GENERATED MINE
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between gap-4">
                  <span className="text-stone-500">Detected Rooms:</span>
                  <span className="font-medium">{roomCount}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-stone-500">Total Area:</span>
                  <span className="font-medium">{totalArea} m²</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-stone-500">Wall Segments:</span>
                  <span className="font-medium">{wallCount}</span>
                </div>
              </div>

              {/* Tracking Status */}
              <div className="mt-4 pt-3 border-t border-stone-200">
                <h4 className="text-xs font-semibold text-orange-600 tracking-wider mb-2">
                  WORKER TRACKING
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between gap-4 items-center">
                    <span className="text-stone-500">Mode:</span>
                    <span className={`font-medium flex items-center gap-1 ${
                      trackingStore.viewMode === 'track' ? 'text-orange-600' : 'text-blue-600'
                    }`}>
                      {trackingStore.viewMode === 'track' ? (
                        <><User size={12} /> Track</>
                      ) : (
                        <><Eye size={12} /> Explore</>
                      )}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4 items-center">
                    <span className="text-stone-500">Connection:</span>
                    <span className={`font-medium flex items-center gap-1 ${
                      trackingStore.connectionStatus === 'connected' ? 'text-green-600' :
                      trackingStore.connectionStatus === 'simulating' ? 'text-yellow-600' : 'text-red-500'
                    }`}>
                      {trackingStore.connectionStatus === 'connected' ? (
                        <><Wifi size={12} /> Live (Real Data)</>
                      ) : trackingStore.connectionStatus === 'simulating' ? (
                        <><Radio size={12} /> Simulating</>
                      ) : (
                        <><WifiOff size={12} /> Offline</>
                      )}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-stone-500">Steps:</span>
                    <span className="font-medium">{trackingStore.stepCount}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-stone-500">Position:</span>
                    <span className="font-medium font-mono text-xs">
                      ({trackingStore.workerPosition.x.toFixed(1)}, {trackingStore.workerPosition.z.toFixed(1)})
                    </span>
                  </div>
                </div>

                {/* Mode Toggle and Reset */}
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={() => {
                      const newMode = trackingStore.viewMode === 'track' ? 'explore' : 'track';
                      trackingStore.setViewMode(newMode);
                      trackingStore.setIsTracking(newMode === 'track');
                    }}
                    className={`flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      trackingStore.viewMode === 'track'
                        ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                        : 'bg-orange-100 text-orange-700 hover:bg-orange-200'
                    }`}
                  >
                    {trackingStore.viewMode === 'track' ? (
                      <><Eye size={12} /> Switch to Explore</>
                    ) : (
                      <><User size={12} /> Switch to Track</>
                    )}
                  </button>
                  <button
                    onClick={() => trackingStore.resetWorkerPosition()}
                    className="flex items-center justify-center gap-1 px-3 py-1.5 bg-stone-100 text-stone-700 rounded-lg text-xs font-medium hover:bg-stone-200 transition-colors"
                    title="Reset worker to entrance"
                  >
                    <RotateCcw size={12} />
                  </button>
                </div>

                {/* Simulation & 2D Map Controls */}
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={() => isSimulating ? stopSimulation() : startSimulation()}
                    className={`flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      isSimulating
                        ? 'bg-red-100 text-red-700 hover:bg-red-200'
                        : 'bg-green-100 text-green-700 hover:bg-green-200'
                    }`}
                  >
                    {isSimulating ? (
                      <><Radio size={12} className="animate-pulse" /> Stop Sim</>
                    ) : (
                      <><Play size={12} /> Demo Sim</>
                    )}
                  </button>
                  <button
                    onClick={() => setShow2DMap(!show2DMap)}
                    className={`flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      show2DMap
                        ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                        : 'bg-stone-100 text-stone-700 hover:bg-stone-200'
                    }`}
                  >
                    <Map size={12} /> {show2DMap ? 'Hide 2D' : 'Show 2D'}
                  </button>
                </div>
              </div>
            </div>

            {/* Minimap */}
            <div className="absolute top-4 right-4 bg-black/50 rounded-xl p-2 shadow-lg">
              <div className="text-xs font-semibold text-white tracking-wider mb-2 flex items-center gap-2">
                <Map size={14} />
                MINIMAP
              </div>
              <canvas
                ref={minimapCanvasRef}
                width={180}
                height={180}
                className="rounded-lg"
              />
              {/* Legend */}
              <div className="mt-2 flex items-center justify-between text-xs text-white/70">
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                  <span>Worker</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span>Gate</span>
                </div>
                {trackingStore.viewMode === 'explore' && (
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    <span>Camera</span>
                  </div>
                )}
              </div>
            </div>

            {/* Action buttons */}
            <div className="absolute bottom-4 left-4 flex gap-2 pointer-events-auto">
              <button
                onClick={backToUpload}
                className="flex items-center gap-2 px-4 py-2.5 bg-white rounded-lg shadow-lg hover:bg-stone-50 transition-colors"
              >
                <ArrowLeft size={18} className="text-orange-500" />
                New Blueprint
              </button>
              <button
                onClick={() => document.documentElement.requestFullscreen()}
                className="flex items-center gap-2 px-4 py-2.5 bg-white rounded-lg shadow-lg hover:bg-stone-50 transition-colors"
              >
                <Maximize2 size={18} className="text-orange-500" />
                Fullscreen
              </button>
              <button
                onClick={exportMineData}
                className="flex items-center gap-2 px-4 py-2.5 bg-white rounded-lg shadow-lg hover:bg-stone-50 transition-colors"
              >
                <Download size={18} className="text-orange-500" />
                Export JSON
              </button>
            </div>

            {/* Controls help - changes based on mode */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-white px-4 py-2 rounded-full shadow-lg text-xs text-stone-500">
              {trackingStore.viewMode === 'track' ? (
                'Camera follows worker | F: Flashlight | Worker position from helmet IMU'
              ) : (
                'WASD/Arrow Keys: Move | Mouse: Look | Shift: Run | F: Flashlight | Click to start'
              )}
            </div>

            {/* Crosshair - only in explore mode */}
            {trackingStore.viewMode === 'explore' && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-2xl text-orange-500 opacity-70">
                +
              </div>
            )}

            {/* 2D Map Overlay - Full screen tracking view */}
            {show2DMap && (
              <div className="absolute inset-0 bg-slate-900/95 backdrop-blur-sm flex items-center justify-center pointer-events-auto">
                <div className="relative bg-slate-800 rounded-2xl shadow-2xl border border-slate-600 p-6 max-w-4xl w-full mx-4">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-orange-500/20 flex items-center justify-center">
                        <Map size={20} className="text-orange-400" />
                      </div>
                      <div>
                        <h2 className="text-lg font-bold text-white">2D Geolocation Tracking</h2>
                        <p className="text-xs text-slate-400">Real-time worker position from helmet IMU</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {/* Status indicator */}
                      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                        isSimulating ? 'bg-yellow-500/20 text-yellow-400' :
                        trackingStore.connectionStatus === 'connected' ? 'bg-green-500/20 text-green-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        <div className={`w-2 h-2 rounded-full ${
                          isSimulating ? 'bg-yellow-400 animate-pulse' :
                          trackingStore.connectionStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                        }`} />
                        {isSimulating ? 'Simulating' :
                         trackingStore.connectionStatus === 'connected' ? 'Live' : 'Offline'}
                      </div>
                      <button
                        onClick={() => setShow2DMap(false)}
                        className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                      >
                        <ArrowLeft size={20} className="text-slate-400" />
                      </button>
                    </div>
                  </div>

                  {/* 2D Map Canvas */}
                  <div className="relative bg-slate-900 rounded-xl overflow-hidden">
                    <canvas
                      ref={map2DCanvasRef}
                      width={800}
                      height={500}
                      className="w-full"
                    />
                  </div>

                  {/* Stats bar */}
                  <div className="flex items-center justify-between mt-4 px-2">
                    <div className="flex gap-6">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-400">{trackingStore.stepCount}</div>
                        <div className="text-xs text-slate-400">Steps</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-400">
                          {(Math.sqrt(
                            Math.pow(trackingStore.workerPosition.x - trackingStore.entrancePosition.x, 2) +
                            Math.pow(trackingStore.workerPosition.z - trackingStore.entrancePosition.z, 2)
                          )).toFixed(1)}m
                        </div>
                        <div className="text-xs text-slate-400">From Gate</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-400">
                          {((trackingStore.workerHeading * 180 / Math.PI) % 360).toFixed(0)}°
                        </div>
                        <div className="text-xs text-slate-400">Heading</div>
                      </div>
                    </div>

                    {/* Quick controls */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => isSimulating ? stopSimulation() : startSimulation()}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                          isSimulating
                            ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                            : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                        }`}
                      >
                        {isSimulating ? <><Radio size={16} className="animate-pulse" /> Stop</> : <><Play size={16} /> Simulate</>}
                      </button>
                      <button
                        onClick={() => trackingStore.resetWorkerPosition()}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-700 text-slate-300 rounded-lg text-sm font-medium hover:bg-slate-600 transition-colors"
                      >
                        <RotateCcw size={16} /> Reset
                      </button>
                    </div>
                  </div>

                  {/* Legend */}
                  <div className="flex items-center gap-6 mt-4 pt-4 border-t border-slate-700 text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full bg-orange-500" />
                      <span>Worker Position</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full bg-green-500" />
                      <span>Gate/Entrance</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-green-400" />
                      <span>Walls</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-green-500/20 border border-green-500/30" />
                      <span>Rooms</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Upload Panel */
        <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center p-6">
          <div className="max-w-4xl w-full">
            {/* Header */}
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-orange-400 to-amber-400 mb-4">
                <Map className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-orange-600 mb-2">Blueprint to 3D Mine Converter</h1>
              <p className="text-stone-500">Upload a 2D mine blueprint to generate a navigable 3D environment</p>
            </div>

            {/* Drop Zone */}
            {!showPreview && (
              <div
                ref={dropZoneRef}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className="border-2 border-dashed border-stone-300 rounded-2xl p-16 text-center bg-white cursor-pointer hover:border-orange-400 hover:bg-orange-50/50 transition-all"
              >
                <Upload className="w-16 h-16 text-orange-400 mx-auto mb-4" />
                <p className="text-lg text-stone-700 mb-2">Drag & drop your blueprint image here</p>
                <span className="text-stone-400 text-sm">or</span>
                <div className="mt-4">
                  <span className="inline-block px-6 py-2.5 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-full font-medium hover:shadow-lg transition-all">
                    Browse Files
                  </span>
                </div>
                <p className="text-stone-400 text-sm mt-4">Supports: PNG, JPG, JPEG (max 10MB)</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  className="hidden"
                />
              </div>
            )}

            {/* Preview Section */}
            {showPreview && (
              <div className="space-y-6">
                {/* Preview Header */}
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold text-orange-600">Blueprint Preview</h3>
                  <button
                    onClick={() => {
                      setShowPreview(false);
                      setUploadedImage(null);
                      setProcessedData(null);
                    }}
                    className="px-4 py-2 border border-stone-200 rounded-lg text-sm hover:bg-stone-50 transition-colors"
                  >
                    Change File
                  </button>
                </div>

                {/* Preview Grid */}
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-white border border-stone-200 rounded-xl p-4">
                    <h4 className="text-xs uppercase tracking-wider text-stone-400 mb-3">Original</h4>
                    <canvas ref={originalCanvasRef} className="max-w-full rounded-lg bg-stone-50" />
                  </div>
                  <div className="bg-white border border-stone-200 rounded-xl p-4">
                    <h4 className="text-xs uppercase tracking-wider text-stone-400 mb-3">Detected Walls</h4>
                    <canvas ref={processedCanvasRef} className="max-w-full rounded-lg bg-stone-50" />
                  </div>
                </div>

                {/* Settings */}
                <div className="bg-white border border-stone-200 rounded-xl p-5">
                  <h3 className="text-sm font-semibold text-orange-600 uppercase tracking-wider mb-4 pb-3 border-b flex items-center gap-2">
                    <Settings2 size={16} />
                    Processing Settings
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center gap-4">
                      <label className="w-48 text-sm font-medium text-stone-700">Wall Detection Threshold</label>
                      <input
                        type="range"
                        min="50"
                        max="200"
                        value={threshold}
                        onChange={(e) => setThreshold(Number(e.target.value))}
                        className="flex-1 accent-orange-500"
                      />
                      <span className="w-12 text-right text-orange-600 font-semibold">{threshold}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <label className="w-48 text-sm font-medium text-stone-700">Wall Height (meters)</label>
                      <input
                        type="range"
                        min="2"
                        max="8"
                        step="0.5"
                        value={wallHeight}
                        onChange={(e) => setWallHeight(Number(e.target.value))}
                        className="flex-1 accent-orange-500"
                      />
                      <span className="w-12 text-right text-orange-600 font-semibold">{wallHeight}m</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <label className="w-48 text-sm font-medium text-stone-700">Scale Factor</label>
                      <input
                        type="range"
                        min="0.5"
                        max="3"
                        step="0.1"
                        value={scale}
                        onChange={(e) => setScale(Number(e.target.value))}
                        className="flex-1 accent-orange-500"
                      />
                      <span className="w-12 text-right text-orange-600 font-semibold">{scale}x</span>
                    </div>
                    <div className="flex gap-6 pt-2">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={addLights}
                          onChange={(e) => setAddLights(e.target.checked)}
                          className="w-4 h-4 accent-orange-500"
                        />
                        <span className="text-sm text-stone-700">Auto-generate lighting</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={addDust}
                          onChange={(e) => setAddDust(e.target.checked)}
                          className="w-4 h-4 accent-orange-500"
                        />
                        <span className="text-sm text-stone-700">Add dust particles</span>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-orange-50 border border-orange-200 rounded-xl p-4 text-center">
                    <div className="text-2xl font-bold text-orange-600">{wallCount}</div>
                    <div className="text-xs text-orange-500 uppercase tracking-wider">Wall Segments</div>
                  </div>
                  <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-center">
                    <div className="text-2xl font-bold text-amber-600">{roomCount}</div>
                    <div className="text-xs text-amber-500 uppercase tracking-wider">Rooms Detected</div>
                  </div>
                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 text-center">
                    <div className="text-2xl font-bold text-yellow-600">{totalArea}</div>
                    <div className="text-xs text-yellow-600 uppercase tracking-wider">Total Area (m²)</div>
                  </div>
                </div>

                {/* Generate Button */}
                <button
                  onClick={generate3DEnvironment}
                  disabled={!processedData || !threeReady}
                  className="w-full py-4 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-xl font-semibold text-lg flex items-center justify-center gap-2 hover:shadow-lg hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play size={20} />
                  {!threeReady ? 'Loading 3D Engine...' : !processedData ? 'Processing Blueprint...' : 'Generate 3D Environment'}
                  <ChevronRight size={20} />
                </button>

                {/* Loading status indicator */}
                {(!threeReady || !cvReady) && (
                  <p className="text-center text-sm text-stone-400 mt-2">
                    {!threeReady && !cvReady ? 'Loading Three.js and OpenCV...' :
                     !threeReady ? 'Loading Three.js...' :
                     !cvReady ? 'Loading OpenCV...' : ''}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </AppLayout>
  );
}
