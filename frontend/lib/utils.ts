import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const getWasteTypeColor = (wasteType: string) => {
  const colors: Record<string, string> = {
    cardboard: "text-amber-400",
    glass: "text-cyan-400",
    metal: "text-gray-400",
    paper: "text-blue-400",
    plastic: "text-red-400",
    trash: "text-orange-400",
  };
  return colors[wasteType.toLowerCase()] || "text-zinc-300";
};

export const getWasteTypeEmoji = (wasteType: string) => {
  const emojis: Record<string, string> = {
    cardboard: "📦",
    glass: "🥃",
    metal: "🔧",
    paper: "📄",
    plastic: "🥤",
    trash: "🗑️",
  };
  return emojis[wasteType.toLowerCase()] || "♻️";
};
