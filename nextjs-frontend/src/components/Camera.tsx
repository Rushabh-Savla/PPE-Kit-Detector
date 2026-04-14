'use client';

import { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera as CameraIcon, Upload, RefreshCw } from 'lucide-react';
import { dataURLtoFile } from '@/lib/utils';

interface CameraProps {
  onCapture: (file: File) => void;
  disabled?: boolean;
}

export default function Camera({ onCapture, disabled = false }: CameraProps) {
  const webcamRef = useRef<Webcam>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [useCamera, setUseCamera] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const capture = useCallback(() => {
    if (webcamRef.current) {
      // Use explicit dimensions for higher quality screenshot (matching video constraints)
      const imageSrc = webcamRef.current.getScreenshot({
        width: 1280,
        height: 720,
      });
      if (imageSrc) {
        setCapturedImage(imageSrc);
        const file = dataURLtoFile(imageSrc, 'capture.png');
        onCapture(file);
      }
    }
  }, [onCapture]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setCapturedImage(result);
      };
      reader.readAsDataURL(file);
      onCapture(file);
    }
  };

  const resetCapture = () => {
    setCapturedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleCameraError = () => {
    setCameraError('Unable to access camera. Please check permissions or use file upload.');
    setUseCamera(false);
  };

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
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${useCamera
              ? 'bg-orange-600 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
            }`}
        >
          <CameraIcon className="inline-block mr-2" size={18} />
          Camera
        </button>
        <button
          onClick={() => {
            setUseCamera(false);
            resetCapture();
          }}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${!useCamera
              ? 'bg-orange-600 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
            }`}
        >
          <Upload className="inline-block mr-2" size={18} />
          Upload
        </button>
      </div>

      {/* Camera or Upload area */}
      <div className="relative aspect-video bg-stone-900 rounded-lg overflow-hidden">
        {capturedImage ? (
          <img
            src={capturedImage}
            alt="Captured"
            className="w-full h-full object-contain"
          />
        ) : useCamera ? (
          cameraError ? (
            <div className="flex items-center justify-center h-full text-white text-center p-4">
              <p>{cameraError}</p>
            </div>
          ) : (
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/png"
              mirrored={false}
              videoConstraints={{
                width: 1280,
                height: 720,
                facingMode: 'user',
              }}
              onUserMediaError={handleCameraError}
              className="w-full h-full object-cover"
            />
          )
        ) : (
          <label className="flex flex-col items-center justify-center h-full cursor-pointer hover:bg-stone-800 transition-colors">
            <Upload size={48} className="text-stone-500 mb-2" />
            <span className="text-stone-400">Click to upload an image</span>
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

      {/* Action buttons */}
      <div className="flex gap-2">
        {useCamera && !capturedImage && !cameraError && (
          <button
            onClick={capture}
            disabled={disabled}
            className="flex-1 py-3 px-4 bg-orange-600 text-white rounded-lg font-medium hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <CameraIcon className="inline-block mr-2" size={18} />
            Capture Photo
          </button>
        )}
        {capturedImage && (
          <button
            onClick={resetCapture}
            disabled={disabled}
            className="flex-1 py-3 px-4 bg-stone-200 text-stone-700 rounded-lg font-medium hover:bg-stone-300 transition-colors disabled:opacity-50"
          >
            <RefreshCw className="inline-block mr-2" size={18} />
            Retake / Reset
          </button>
        )}
      </div>
    </div>
  );
}
