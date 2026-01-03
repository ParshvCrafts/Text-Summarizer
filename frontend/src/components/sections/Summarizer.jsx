import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Copy, Check, Trash2, SkipForward, Clock, Minimize2 } from 'lucide-react';
import { summarizeText } from '../../utils/api';
import { useTypewriter } from '../../hooks/useTypewriter';
import ConfidenceMeter from '../ui/ConfidenceMeter';
import NeuralNetwork from '../ui/NeuralNetwork';

const EXAMPLE_DIALOGUE = `John: Hey, are you coming to the party tonight?
Sarah: I'm not sure, I have a lot of work to do.
John: Come on, it'll be fun! Everyone's going to be there.
Sarah: Okay, I'll try to finish early and come by around 8.
John: Great! See you then!`;

const Summarizer = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [targetLength, setTargetLength] = useState('medium');
  const [copied, setCopied] = useState(false);
  const [showTypewriter, setShowTypewriter] = useState(true);
  const [processingTime, setProcessingTime] = useState(null);
  const [showCopyToast, setShowCopyToast] = useState(false);
  const startTimeRef = useRef(null);

  const { displayedText, isComplete, skip } = useTypewriter(
    result?.summary || '',
    15,
    showTypewriter && result !== null
  );

  const handleSummarize = useCallback(async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setShowTypewriter(true);
    setProcessingTime(null);
    startTimeRef.current = Date.now();

    try {
      const response = await summarizeText(input, targetLength);
      const endTime = Date.now();
      setProcessingTime(((endTime - startTimeRef.current) / 1000).toFixed(1));
      setResult(response);
    } catch (err) {
      setError(err.message || 'Failed to generate summary. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [input, targetLength]);

  const handleCopy = async () => {
    if (result?.summary) {
      await navigator.clipboard.writeText(result.summary);
      setCopied(true);
      setShowCopyToast(true);
      setTimeout(() => {
        setCopied(false);
        setShowCopyToast(false);
      }, 2000);
    }
  };

  const handleClear = () => {
    setInput('');
    setResult(null);
    setError(null);
  };

  const handleSkip = () => {
    setShowTypewriter(false);
    skip();
  };

  const loadExample = () => {
    setInput(EXAMPLE_DIALOGUE);
    setResult(null);
    setError(null);
  };

  const lengthOptions = [
    { value: 'short', label: 'Short' },
    { value: 'medium', label: 'Medium' },
    { value: 'long', label: 'Long' },
  ];

  return (
    <section id="demo" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="section-title">
            <span className="gradient-text">Try It Yourself</span>
          </h2>
          <p className="section-subtitle">
            Paste a dialogue below and watch the AI transform it into a concise summary
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="glass-card"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">Input Dialogue</h3>
              <div className="flex gap-2">
                <button
                  onClick={loadExample}
                  className="text-sm text-primary hover:text-accent transition-colors"
                >
                  Load Example
                </button>
                {input && (
                  <button
                    onClick={handleClear}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <Trash2 size={18} />
                  </button>
                )}
              </div>
            </div>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Paste your dialogue here..."
              className="input-field min-h-[250px] resize-none font-mono text-sm"
            />

            <div className="flex justify-between items-center mt-4 text-sm text-gray-500">
              <span>{input.length} characters</span>
            </div>

            {/* Length Control */}
            <div className="mt-6">
              <label className="text-sm text-gray-400 mb-2 block">Summary Length</label>
              <div className="flex gap-2">
                {lengthOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setTargetLength(option.value)}
                    className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all ${
                      targetLength === option.value
                        ? 'bg-primary text-white'
                        : 'bg-dark-card text-gray-400 hover:text-white border border-dark-border'
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Summarize Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSummarize}
              disabled={!input.trim() || isLoading}
              className="btn-primary w-full mt-6 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  >
                    <Sparkles size={20} />
                  </motion.div>
                  Processing...
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  Summarize
                </>
              )}
            </motion.button>
          </motion.div>

          {/* Output Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="glass-card"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">Summary</h3>
              {result && (
                <div className="flex gap-2">
                  {!isComplete && (
                    <button
                      onClick={handleSkip}
                      className="text-gray-400 hover:text-white transition-colors"
                      title="Skip animation"
                    >
                      <SkipForward size={18} />
                    </button>
                  )}
                  <button
                    onClick={handleCopy}
                    className="text-gray-400 hover:text-white transition-colors"
                    title="Copy summary"
                  >
                    {copied ? <Check size={18} className="text-success" /> : <Copy size={18} />}
                  </button>
                </div>
              )}
            </div>

            <div className="min-h-[250px] flex items-center justify-center">
              <AnimatePresence mode="wait">
                {isLoading ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center"
                  >
                    <NeuralNetwork isProcessing={true} />
                    <p className="text-gray-400 mt-4">AI is thinking...</p>
                  </motion.div>
                ) : error ? (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center text-error"
                  >
                    <p>{error}</p>
                    <button
                      onClick={handleSummarize}
                      className="mt-4 text-primary hover:text-accent transition-colors"
                    >
                      Try Again
                    </button>
                  </motion.div>
                ) : result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full"
                  >
                    <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
                      <p className="text-white leading-relaxed">
                        {displayedText}
                        {!isComplete && (
                          <motion.span
                            animate={{ opacity: [1, 0] }}
                            transition={{ duration: 0.5, repeat: Infinity }}
                            className="inline-block w-2 h-5 bg-primary ml-1 align-middle"
                          />
                        )}
                      </p>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center text-gray-500"
                  >
                    <NeuralNetwork isProcessing={false} />
                    <p className="mt-4">Your summary will appear here</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Copy Toast */}
            <AnimatePresence>
              {showCopyToast && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="absolute top-4 right-4 px-4 py-2 bg-success/20 border border-success/50 rounded-lg text-success text-sm flex items-center gap-2"
                >
                  <Check size={16} />
                  Copied to clipboard!
                </motion.div>
              )}
            </AnimatePresence>

            {/* Metrics */}
            {result && isComplete && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 pt-6 border-t border-dark-border"
              >
                <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
                  <ConfidenceMeter value={result.confidence} size={100} />
                  
                  <div className="flex-1 space-y-4">
                    {/* Compression Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400 flex items-center gap-1">
                          <Minimize2 size={14} />
                          Compression
                        </span>
                        <span className="text-primary font-bold">
                          {Math.round((1 - result.summary_length / result.input_length) * 100)}% shorter
                        </span>
                      </div>
                      <div className="h-3 bg-dark-card rounded-full overflow-hidden border border-dark-border">
                        <motion.div
                          initial={{ width: '100%' }}
                          animate={{ width: `${(result.summary_length / result.input_length) * 100}%` }}
                          transition={{ duration: 1, ease: 'easeOut' }}
                          className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                        />
                      </div>
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>{result.input_length} chars input</span>
                        <span>{result.summary_length} chars output</span>
                      </div>
                    </div>

                    {/* Processing Time */}
                    {processingTime && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center justify-center gap-2 text-sm text-gray-400"
                      >
                        <Clock size={14} className="text-primary" />
                        <span>Generated in <span className="text-white font-medium">{processingTime}s</span></span>
                      </motion.div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Summarizer;
