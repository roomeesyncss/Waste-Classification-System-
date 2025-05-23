import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ImageUploadDetection from "@/components/image-upload-detection";
import CameraStreamDetection from "@/components/camera-stream-detection";

export default function Home() {
  return (
    <main className="min-h-screen p-6 bg-gradient-to-br from-zinc-950 via-zinc-900 to-black">
      <div className="container mx-auto w-full flex flex-col items-center justify-center">
        <div className="max-w-4xl mx-auto relative w-full">
          <div className="absolute -top-10 -left-10 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl opacity-20"></div>
          <div className="absolute -bottom-10 -right-10 w-64 h-64 bg-teal-500/10 rounded-full blur-3xl opacity-20"></div>

          <div className="relative w-full pt-10">
            <h1 className="text-4xl font-bold text-center mb-2 mt-8 text-white tracking-tight">
              Waste Classifier
            </h1>
            <p className="text-zinc-400 text-center mb-10 max-w-lg mx-auto">
              Upload images or use your camera to detect objects in real-time
              with our advanced AI system
            </p>

            <div className="w-full bg-zinc-900/80 backdrop-blur-sm rounded-xl p-8 shadow-2xl border border-zinc-800/50 relative overflow-hidden">
              <div className="absolute inset-0 bg-grid-white/[0.02] -z-10"></div>

              <Tabs defaultValue="image" className="w-full">
                <TabsList className="grid w-full grid-cols-2 mb-8 bg-zinc-800/50 sm:p-1 rounded-lg">
                  <TabsTrigger
                    value="image"
                    className="data-[state=active]:bg-zinc-700 data-[state=active]:text-white rounded-md transition-all"
                  >
                    Image Upload
                  </TabsTrigger>
                  <TabsTrigger
                    value="camera"
                    className="data-[state=active]:bg-zinc-700 data-[state=active]:text-white rounded-md transition-all"
                  >
                    Camera Stream
                  </TabsTrigger>
                </TabsList>

                <TabsContent
                  value="image"
                  className="animate-in fade-in-50 duration-300"
                >
                  <ImageUploadDetection />
                </TabsContent>

                <TabsContent
                  value="camera"
                  className="animate-in fade-in-50 duration-300"
                >
                  <CameraStreamDetection />
                </TabsContent>
              </Tabs>
            </div>

            <footer className="mt-8 text-center text-zinc-500 text-sm">
              <p>© 2025 Object Detection System • AI CCP Project</p>
            </footer>
          </div>
        </div>
      </div>
    </main>
  );
}
