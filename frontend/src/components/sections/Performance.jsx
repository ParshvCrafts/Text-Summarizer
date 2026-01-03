import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const metrics = [
  { label: 'ROUGE-1 Score', value: 43.53, suffix: '', color: 'text-primary' },
  { label: 'ROUGE-2 Score', value: 20.01, suffix: '', color: 'text-accent' },
  { label: 'ROUGE-L Score', value: 34.78, suffix: '', color: 'text-secondary' },
  { label: 'Model Parameters', value: 248, suffix: 'M', color: 'text-white' },
];

const AnimatedNumber = ({ value, suffix = '', duration = 2 }) => {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = value;
    const increment = end / (duration * 60);
    let current = start;

    const timer = setInterval(() => {
      current += increment;
      if (current >= end) {
        setDisplay(end);
        clearInterval(timer);
      } else {
        setDisplay(current);
      }
    }, 1000 / 60);

    return () => clearInterval(timer);
  }, [value, duration]);

  return (
    <span>
      {display.toFixed(suffix === 'M' ? 0 : 2)}{suffix}
    </span>
  );
};

const Performance = () => {
  return (
    <section className="py-20 px-4 bg-gradient-to-b from-transparent via-dark-card/30 to-transparent">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">
            <span className="gradient-text">Benchmark Results</span>
          </h2>
          <p className="section-subtitle">
            Evaluated on the SAMSum test dataset with industry-standard ROUGE metrics
          </p>
        </motion.div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card text-center"
            >
              <motion.div
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                className={`text-4xl lg:text-5xl font-bold ${metric.color} mb-2`}
              >
                <AnimatedNumber value={metric.value} suffix={metric.suffix} />
              </motion.div>
              <div className="text-sm text-gray-400">{metric.label}</div>
            </motion.div>
          ))}
        </div>

        {/* Additional info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-12 glass-card"
        >
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-2xl font-bold text-white mb-1">FLAN-T5 Base</div>
              <div className="text-sm text-gray-400">Model Architecture</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white mb-1">SAMSum</div>
              <div className="text-sm text-gray-400">Training Dataset</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white mb-1">~3 seconds</div>
              <div className="text-sm text-gray-400">Average Inference Time</div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Performance;
