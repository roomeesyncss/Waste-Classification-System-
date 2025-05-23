"use client";

import type React from "react";

import { useState } from "react";
import { Upload, ImageIcon, Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn, getWasteTypeColor, getWasteTypeEmoji } from "@/lib/utils";
import axios from "axios";
import { toast } from "sonner";

const API_BASE_URL = "https://6275-147-135-15-16.ngrok-free.app";

interface PredictionResponse {
  prediction: string;
  filename: string;
}

export default function ImageUploadDetection() {
  const [image, setImage] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] = useState<string | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        // 10 MB limit
        toast.error("File size must be less than 10MB");
        return;
      }

      if (!file.type.startsWith("image/")) {
        toast.error("Please select a valid image file");
        return;
      }

      setFileName(file.name);
      setSelectedFile(file);

      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
        setDetectionResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDetection = async () => {
    if (!selectedFile) {
      toast.error("Please select an image first");
      return;
    }

    setIsProcessing(true);
    setDetectionResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const { data } = await axios.post<PredictionResponse>(
        `${API_BASE_URL}/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 30000,
        }
      );

      setDetectionResult(data.prediction);
      toast.success(`Detected: ${data.prediction}`);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response) {
          const status = error.response.status;
          const message = error.response.data?.detail || "Detection failed";

          if (status === 400) toast.error(`Invalid request: ${message}`);
          else toast.error(`Server error: ${message}`);
        } else toast.error("An unexpected error occurred");
      } else toast.error("An unexpected error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  const clearImage = () => {
    setImage(null);
    setFileName("");
    setSelectedFile(null);
    setDetectionResult(null);

    const fileInput = document.getElementById(
      "image-upload"
    ) as HTMLInputElement;
    if (fileInput) {
      fileInput.value = "";
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center justify-center">
        <label
          htmlFor="image-upload"
          className={cn(
            "border-2 border-dashed rounded-lg p-8 w-full max-w-xl cursor-pointer",
            "flex flex-col items-center justify-center gap-2 transition-colors",
            "hover:border-zinc-700 hover:bg-zinc-800",
            image ? "border-zinc-600" : "border-zinc-800"
          )}
        >
          {image ? (
            <div className="space-y-4 w-full">
              <div className="relative">
                <img
                  src={image}
                  alt="Uploaded"
                  className="max-h-64 mx-auto object-contain rounded-md"
                />
                <Button
                  onClick={(e) => {
                    e.preventDefault();
                    clearImage();
                  }}
                  variant="destructive"
                  size="sm"
                  className="absolute top-2 right-2 h-8 w-8 p-0"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-sm text-center text-gray-500 truncate px-4">
                {fileName}
              </p>
            </div>
          ) : (
            <>
              <Upload className="h-10 w-10 text-gray-400" />
              <p className="text-sm font-medium">Click to upload an image</p>
              <p className="text-xs text-zinc-400">PNG, JPG, JPEG up to 10MB</p>
              <p className="text-xs text-zinc-500 mt-1">
                Supported: Cardboard, Glass, Metal, Paper, Plastic, Trash
              </p>
            </>
          )}
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />
        </label>
      </div>

      <div className="flex justify-center gap-3">
        <Button
          onClick={handleDetection}
          disabled={!selectedFile || isProcessing}
          className="flex-1 max-w-sm cursor-pointer bg-purple-800 hover:bg-purple-700 text-white transition-colors disabled:opacity-50"
        >
          {isProcessing ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Classifying...
            </>
          ) : (
            <>
              <ImageIcon className="mr-2 h-4 w-4" />
              Classify Waste
            </>
          )}
        </Button>

        {image && (
          <Button
            onClick={clearImage}
            variant="outline"
            className="px-4 border-zinc-700 hover:bg-zinc-800"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </div>

      {detectionResult && (
        <Card className="mt-6 bg-zinc-900/80 border-zinc-800/50 overflow-hidden backdrop-blur-sm">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 to-teal-500/5"></div>
          <CardContent className="pt-6 relative">
            <div className="flex items-center mb-4">
              <div className="w-2 h-2 rounded-full bg-purple-500 mr-2 animate-pulse"></div>
              <h3 className="text-lg font-medium text-zinc-200">
                Classification Result
              </h3>
            </div>
            <div className="p-6 bg-zinc-800/80 rounded-lg border border-zinc-700/50 shadow-inner">
              <div className="flex items-center justify-center space-x-3">
                <span className="text-3xl">
                  {getWasteTypeEmoji(detectionResult)}
                </span>
                <div className="text-center">
                  <p
                    className={cn(
                      "text-2xl font-bold capitalize",
                      getWasteTypeColor(detectionResult)
                    )}
                  >
                    {detectionResult}
                  </p>
                  <p className="text-sm text-zinc-400 mt-1">
                    Waste Type Detected
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
