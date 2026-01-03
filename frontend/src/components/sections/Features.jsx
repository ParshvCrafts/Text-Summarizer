import { motion } from 'framer-motion';
import { Zap, Shield, Sliders, MessageSquare, Gauge, Clock } from 'lucide-react';

const features = [
  {
    icon: Zap,
    title: 'Instant Summarization',
    description: 'Get concise summaries in just 3-4 seconds using state-of-the-art transformer models.',
    color: 'from-yellow-500 to-orange-500',
  },
  {
    icon: Gauge,
    title: 'Confidence Scoring',
    description: 'Each summary includes a confidence score so you know how reliable the output is.',
    color: 'from-green-500 to-emerald-500',
  },
  {
    icon: Sliders,
    title: 'Length Control',
    description: 'Choose between short, medium, or long summaries based on your needs.',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: MessageSquare,
    title: 'Dialogue Understanding',
    description: 'Trained on SAMSum dataset specifically for conversation summarization.',
    color: 'from-purple-500 to-pink-500',
  },
  {
    icon: Clock,
    title: 'Fast Processing',
    description: '248M parameter FLAN-T5 model optimized for quick inference times.',
    color: 'from-red-500 to-rose-500',
  },
  {
    icon: Shield,
    title: 'Privacy First',
    description: 'Your data is processed locally. No conversations are stored or logged.',
    color: 'from-indigo-500 to-violet-500',
  },
];

const Features = () => {
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
            <span className="gradient-text">Powerful Features</span>
          </h2>
          <p className="section-subtitle">
            Everything you need for intelligent dialogue summarization
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -5 }}
              className="glass-card group cursor-default"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} p-0.5 mb-4`}>
                <div className="w-full h-full rounded-xl bg-dark-card flex items-center justify-center">
                  <feature.icon size={24} className="text-white" />
                </div>
              </div>

              <h3 className="text-xl font-semibold text-white mb-2 group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
