import { useEffect, useState } from "react";

type ThemeChoice = "system" | "light" | "dark";

function resolveIsDark(choice: ThemeChoice): boolean {
  if (choice === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  return choice === "dark";
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(
    () => (localStorage.getItem("pde-studio-theme") as ThemeChoice) || "system"
  );
  const [isDark, setIsDark] = useState(() => resolveIsDark(choice));

  useEffect(() => {
    localStorage.setItem("pde-studio-theme", choice);
    setIsDark(resolveIsDark(choice));
    if (choice === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", choice);
    }
  }, [choice]);

  useEffect(() => {
    if (choice !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => setIsDark(mq.matches);
    mq.addEventListener("change", listener);
    return () => mq.removeEventListener("change", listener);
  }, [choice]);

  const toggle = () => setChoice(isDark ? "light" : "dark");

  return { isDark, toggle };
}
