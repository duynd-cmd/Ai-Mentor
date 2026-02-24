import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

declare global {
  interface Window {
    MathJax?: {
      startup?: { promise?: Promise<unknown> };
      typesetPromise?: (elements?: HTMLElement[]) => Promise<void>;
    };
  }
}

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let isCancelled = false;
    let retryTimer: number | null = null;

    const typesetMath = async (attempt = 0) => {
      const mathJax = window.MathJax;
      if (!containerRef.current) return;
      if (!mathJax?.typesetPromise) {
        if (attempt < 10) {
          retryTimer = window.setTimeout(() => {
            void typesetMath(attempt + 1);
          }, 150);
        }
        return;
      }

      try {
        if (mathJax.startup?.promise) {
          await mathJax.startup.promise;
        }
        if (!isCancelled && containerRef.current) {
          await mathJax.typesetPromise([containerRef.current]);
        }
      } catch (error) {
        // Keep markdown rendering even if MathJax fails.
        console.error('Math typeset error:', error);
      }
    };

    void typesetMath();
    return () => {
      isCancelled = true;
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [content]);

  return (
    <div ref={containerRef} className={className}>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
};
