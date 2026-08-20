"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";

export default function BodyRouteClass() {
  const pathname = usePathname();

  useEffect(() => {
    document.body.classList.toggle("chat-page", pathname === "/");
    return () => document.body.classList.remove("chat-page");
  }, [pathname]);

  return null;
}
