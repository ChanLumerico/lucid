"use client";

import { motion } from "framer-motion";
import { springs } from "@/components/motion/springs";

export default function Template({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={springs.smooth}
    >
      {children}
    </motion.div>
  );
}
