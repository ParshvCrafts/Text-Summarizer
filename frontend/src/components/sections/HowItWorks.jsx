import { motion } from 'framer-motion';
import { MessageSquare, Cpu, Brain, FileText, Sparkles } from 'lucide-react';

const steps = [
  {
    icon: MessageSquare,
    title: 'Input Dialogue',
    description: 'Paste your conversation or dialogue text into the input field.',
  },
  {
    icon: Cpu,
    title: 'Tokenization',
    description: 'Text is converted into tokens that the model can understand.',
  },
  {
    icon: Brain,
    title: 'FLAN-T5 Processing',
    description: 'The 248M parameter model analyzes and compresses the content.',
  },
  {
    icon: FileText,
    title: 'Summary Generation',
    description: 'Beam search generates the most likely summary sequence.',
  },
  {
    icon: Sparkles,
    title: 'Output & Confidence',
    description: 'Receive your summary with a confidence score and metrics.',
  },
];

const HowItWorks = () => {
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
            <span className="gradient-text">How It Works</span>
          </h2>
          <p className="section-subtitle">
            A peek under the hood of the AI summarization pipeline
          </p>
        </motion.div>

        <div className="relative">
          {/* Connection line */}
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-primary via-accent to-secondary opacity-20" />

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-8">
            {steps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="relative"
              >
                <div className="glass-card text-center group hover:border-primary/30 transition-all duration-300">
                  {/* Step number */}
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">
                    {index + 1}
                  </div>

                  {/* Icon */}
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center group-hover:from-primary/30 group-hover:to-secondary/30 transition-all"
                  >
                    <step.icon size={28} className="text-primary" />
                  </motion.div>

                  <h3 className="text-lg font-semibold text-white mb-2">{step.title}</h3>
                  <p className="text-sm text-gray-400">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
