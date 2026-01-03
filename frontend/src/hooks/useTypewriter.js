import { useState, useEffect, useCallback } from 'react';

export const useTypewriter = (text, speed = 20, enabled = true) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (!enabled || !text) {
      setDisplayedText(text || '');
      setIsComplete(true);
      return;
    }

    setDisplayedText('');
    setIsComplete(false);
    let index = 0;

    const timer = setInterval(() => {
      if (index < text.length) {
        setDisplayedText(text.slice(0, index + 1));
        index++;
      } else {
        setIsComplete(true);
        clearInterval(timer);
      }
    }, speed);

    return () => clearInterval(timer);
  }, [text, speed, enabled]);

  const skip = useCallback(() => {
    setDisplayedText(text || '');
    setIsComplete(true);
  }, [text]);

  return { displayedText, isComplete, skip };
};

export default useTypewriter;
