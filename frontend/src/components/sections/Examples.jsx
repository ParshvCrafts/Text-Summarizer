import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, ArrowRight } from 'lucide-react';

const examples = [
  {
    title: 'Casual Conversation',
    input: `John: Hey, are you coming to the party tonight?
Sarah: I'm not sure, I have a lot of work to do.
John: Come on, it'll be fun! Everyone's going to be there.
Sarah: Okay, I'll try to finish early and come by around 8.
John: Great! See you then!`,
    output: 'Sarah is unsure about attending the party due to work, but agrees to try to come by 8 PM.',
  },
  {
    title: 'Business Meeting',
    input: `Alice: Hi Bob, can we schedule a meeting for tomorrow?
Bob: Sure, what time works for you?
Alice: How about 2 PM?
Bob: That works. Should I book the conference room?
Alice: Yes please. We need to discuss the Q4 budget.
Bob: Got it. I'll send the calendar invite.`,
    output: 'Alice and Bob schedule a meeting for 2 PM tomorrow to discuss the Q4 budget. Bob will book the conference room and send a calendar invite.',
  },
  {
    title: 'Technical Support',
    input: `Customer: My laptop won't turn on. I've tried everything.
Support: Have you tried holding the power button for 10 seconds?
Customer: Yes, nothing happens.
Support: Is the charging light on when you plug it in?
Customer: No, there's no light at all.
Support: It sounds like a power adapter issue. Try a different charger.
Customer: I don't have another one.
Support: I'll arrange for a replacement adapter to be sent to you.`,
    output: "Customer's laptop won't turn on and shows no charging light. Support diagnoses it as a power adapter issue and will send a replacement.",
  },
];

const Examples = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const next = () => setCurrentIndex((i) => (i + 1) % examples.length);
  const prev = () => setCurrentIndex((i) => (i - 1 + examples.length) % examples.length);

  const current = examples[currentIndex];

  return (
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">
            <span className="gradient-text">Example Summaries</span>
          </h2>
          <p className="section-subtitle">
            See how the AI transforms different types of conversations
          </p>
        </motion.div>

        <div className="relative">
          {/* Navigation buttons */}
          <button
            onClick={prev}
            className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 lg:-translate-x-12 z-10 w-10 h-10 rounded-full glass flex items-center justify-center text-gray-400 hover:text-white hover:border-primary/50 transition-all"
          >
            <ChevronLeft size={24} />
          </button>
          <button
            onClick={next}
            className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 lg:translate-x-12 z-10 w-10 h-10 rounded-full glass flex items-center justify-center text-gray-400 hover:text-white hover:border-primary/50 transition-all"
          >
            <ChevronRight size={24} />
          </button>

          <AnimatePresence mode="wait">
            <motion.div
              key={currentIndex}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
              className="glass-card"
            >
              <div className="text-center mb-6">
                <span className="text-sm text-primary font-medium">{current.title}</span>
              </div>

              <div className="grid lg:grid-cols-2 gap-8 items-start">
                {/* Input */}
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-3">Input Dialogue</h4>
                  <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
                    <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">
                      {current.input}
                    </pre>
                  </div>
                </div>

                {/* Arrow */}
                <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                  <motion.div
                    animate={{ x: [0, 10, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    <ArrowRight size={24} className="text-primary" />
                  </motion.div>
                </div>

                {/* Output */}
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-3">AI Summary</h4>
                  <div className="bg-gradient-to-br from-primary/10 to-secondary/10 rounded-xl p-4 border border-primary/20">
                    <p className="text-white leading-relaxed">{current.output}</p>
                  </div>
                </div>
              </div>

              {/* Dots indicator */}
              <div className="flex justify-center gap-2 mt-8">
                {examples.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentIndex(index)}
                    className={`w-2 h-2 rounded-full transition-all ${
                      index === currentIndex
                        ? 'bg-primary w-6'
                        : 'bg-gray-600 hover:bg-gray-500'
                    }`}
                  />
                ))}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
};

export default Examples;
