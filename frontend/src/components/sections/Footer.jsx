import { motion } from 'framer-motion';
import { Github, ExternalLink, Heart, Linkedin, Globe } from 'lucide-react';

const Footer = () => {
  const socialLinks = [
    { icon: Github, href: 'https://github.com/ParshvCrafts', label: 'GitHub' },
    { icon: Linkedin, href: 'https://www.linkedin.com/in/parshv-patel-65a90326b/', label: 'LinkedIn' },
    { icon: Globe, href: 'https://personal-website-rtzu.onrender.com/', label: 'Portfolio' },
  ];

  return (
    <footer className="py-12 px-4 border-t border-dark-border">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* Brand */}
          <div>
            <h3 className="text-xl font-bold gradient-text mb-4">Text Summarizer</h3>
            <p className="text-gray-400 text-sm mb-4">
              AI-powered dialogue summarization using state-of-the-art transformer models.
            </p>
            {/* Social Icons */}
            <div className="flex gap-3">
              {socialLinks.map((link) => (
                <motion.a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-9 h-9 rounded-lg bg-dark-card border border-dark-border flex items-center justify-center text-gray-400 hover:text-primary hover:border-primary/50 transition-colors"
                  title={link.label}
                >
                  <link.icon size={18} />
                </motion.a>
              ))}
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/ParshvCrafts"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-primary transition-colors flex items-center gap-2 text-sm group"
                >
                  <Github size={16} className="group-hover:scale-110 transition-transform" />
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/philschmid/flan-t5-base-samsum"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-primary transition-colors flex items-center gap-2 text-sm group"
                >
                  <ExternalLink size={16} className="group-hover:scale-110 transition-transform" />
                  Model Card (HuggingFace)
                </a>
              </li>
              <li>
                <a
                  href="/api/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-primary transition-colors flex items-center gap-2 text-sm group"
                >
                  <ExternalLink size={16} className="group-hover:scale-110 transition-transform" />
                  API Documentation
                </a>
              </li>
              <li>
                <a
                  href="https://personal-website-rtzu.onrender.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-primary transition-colors flex items-center gap-2 text-sm group"
                >
                  <Globe size={16} className="group-hover:scale-110 transition-transform" />
                  Developer Portfolio
                </a>
              </li>
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="text-white font-semibold mb-4">Built With</h4>
            <div className="flex flex-wrap gap-2">
              {[
                { name: 'React 19', color: 'hover:border-cyan-500' },
                { name: 'Tailwind CSS', color: 'hover:border-sky-500' },
                { name: 'Framer Motion', color: 'hover:border-purple-500' },
                { name: 'GSAP', color: 'hover:border-green-500' },
                { name: 'FastAPI', color: 'hover:border-teal-500' },
                { name: 'PyTorch', color: 'hover:border-orange-500' },
                { name: 'FLAN-T5', color: 'hover:border-yellow-500' },
              ].map((tech) => (
                <motion.span
                  key={tech.name}
                  whileHover={{ scale: 1.05, y: -1 }}
                  className={`px-3 py-1 text-xs bg-dark-card border border-dark-border rounded-full text-gray-400 hover:text-white transition-all cursor-default ${tech.color}`}
                >
                  {tech.name}
                </motion.span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="pt-8 border-t border-dark-border text-center"
        >
          <p className="text-gray-500 text-sm flex items-center justify-center gap-1">
            Made with <motion.span whileHover={{ scale: 1.2 }}><Heart size={14} className="text-red-500 fill-red-500" /></motion.span> by{' '}
            <a 
              href="https://personal-website-rtzu.onrender.com/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary hover:text-accent transition-colors"
            >
              Parshv Patel
            </a>
          </p>
          <p className="text-gray-600 text-xs mt-2">
            © {new Date().getFullYear()} Text Summarizer. UC Berkeley • Data Science & ML
          </p>
        </motion.div>
      </div>
    </footer>
  );
};

export default Footer;
