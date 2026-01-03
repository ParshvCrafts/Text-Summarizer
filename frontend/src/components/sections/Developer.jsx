import { useRef, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import { Linkedin, Github, Globe, Mail, GraduationCap, Brain, Award } from 'lucide-react';
import gsap from 'gsap';

const Developer = () => {
  const sectionRef = useRef(null);
  const nameRef = useRef(null);
  const isInView = useInView(sectionRef, { once: true, margin: '-100px' });

  const socialLinks = [
    { 
      icon: Linkedin, 
      href: 'https://www.linkedin.com/in/parshv-patel-65a90326b/', 
      label: 'LinkedIn',
      color: '#0077b5'
    },
    { 
      icon: Github, 
      href: 'https://github.com/ParshvCrafts', 
      label: 'GitHub',
      color: '#333'
    },
    { 
      icon: Globe, 
      href: 'https://personal-website-rtzu.onrender.com/', 
      label: 'Portfolio',
      color: '#06b6d4'
    },
    { 
      icon: Mail, 
      href: 'mailto:p1a2r3s4h5v6@gmail.com', 
      label: 'Email',
      color: '#ea4335'
    },
  ];

  const badges = [
    { icon: GraduationCap, text: 'UC Berkeley', color: 'from-blue-500 to-yellow-500' },
    { icon: Brain, text: 'AI/ML Focus', color: 'from-purple-500 to-pink-500' },
    { icon: Award, text: '4.0 GPA', color: 'from-green-500 to-emerald-500' },
  ];

  useEffect(() => {
    if (isInView && nameRef.current) {
      const chars = nameRef.current.querySelectorAll('.char');
      gsap.fromTo(chars, 
        { opacity: 0, y: 50, rotateX: -90 },
        { 
          opacity: 1, 
          y: 0, 
          rotateX: 0,
          duration: 0.8,
          stagger: 0.05,
          ease: 'back.out(1.7)'
        }
      );
    }
  }, [isInView]);

  const splitText = (text) => {
    return text.split('').map((char, i) => (
      <span key={i} className="char inline-block" style={{ opacity: 0 }}>
        {char === ' ' ? '\u00A0' : char}
      </span>
    ));
  };

  return (
    <section 
      ref={sectionRef}
      className="py-24 px-4 relative overflow-hidden"
      id="about"
    >
      {/* Animated gradient background */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/50 via-purple-900/30 to-yellow-900/20 animate-gradient" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-radial from-primary/20 via-transparent to-transparent rounded-full blur-3xl" />
      </div>

      <div className="max-w-5xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="section-title">
            <span className="gradient-text">About the Developer</span>
          </h2>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-8 items-center">
          {/* Left side - Stylized initials */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="lg:col-span-2 flex justify-center"
          >
            <div className="relative group">
              {/* Glow effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary via-accent to-secondary rounded-3xl blur-2xl opacity-40 group-hover:opacity-60 transition-opacity duration-500 animate-pulse-slow" />
              
              {/* Main initials card */}
              <div className="relative w-48 h-48 md:w-64 md:h-64 rounded-3xl bg-gradient-to-br from-dark-card to-dark-bg border border-dark-border flex items-center justify-center overflow-hidden">
                {/* Animated border */}
                <div className="absolute inset-0 rounded-3xl p-[2px] bg-gradient-to-br from-primary via-secondary to-accent opacity-50">
                  <div className="w-full h-full rounded-3xl bg-dark-card" />
                </div>
                
                {/* Initials */}
                <span className="relative text-7xl md:text-8xl font-bold gradient-text select-none">
                  PP
                </span>
                
                {/* Floating particles */}
                <div className="absolute top-4 right-4 w-2 h-2 bg-primary rounded-full animate-float opacity-60" />
                <div className="absolute bottom-6 left-6 w-3 h-3 bg-secondary rounded-full animate-float opacity-40" style={{ animationDelay: '1s' }} />
                <div className="absolute top-1/3 left-4 w-1.5 h-1.5 bg-accent rounded-full animate-float opacity-50" style={{ animationDelay: '2s' }} />
              </div>
            </div>
          </motion.div>

          {/* Right side - Bio and info */}
          <div className="lg:col-span-3 space-y-6">
            {/* Name with character animation */}
            <div ref={nameRef} className="overflow-hidden">
              <h3 className="text-4xl md:text-5xl font-bold text-white mb-2">
                {splitText('Parshv Patel')}
              </h3>
            </div>

            {/* Title */}
            <motion.p
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="text-xl text-primary font-medium"
            >
              Data Science & Machine Learning Engineer
            </motion.p>

            {/* Badges */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap gap-3"
            >
              {badges.map((badge, index) => (
                <motion.div
                  key={badge.text}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  whileHover={{ scale: 1.05, y: -2 }}
                  className="relative group"
                >
                  <div className={`absolute inset-0 bg-gradient-to-r ${badge.color} rounded-full blur opacity-40 group-hover:opacity-60 transition-opacity`} />
                  <div className="relative flex items-center gap-2 px-4 py-2 bg-dark-card border border-dark-border rounded-full">
                    <badge.icon size={16} className="text-white" />
                    <span className="text-sm font-medium text-white">{badge.text}</span>
                  </div>
                  {badge.text === '4.0 GPA' && (
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer" />
                    </div>
                  )}
                </motion.div>
              ))}
            </motion.div>

            {/* Bio */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6 }}
              className="glass-card"
            >
              <p className="text-gray-300 leading-relaxed mb-4">
                I'm a freshman at <span className="text-primary font-medium">UC Berkeley</span> pursuing a B.A. in Data Science with a focus on AI and Machine Learning. With a perfect 4.0 GPA and a passion for transforming data into actionable insights, I specialize in building end-to-end machine learning solutions.
              </p>
              <p className="text-gray-400 leading-relaxed">
                From recommendation systems to performance analytics, I love creating projects that solve real-world problems. My work spans data wrangling, statistical modeling, deep learning, and full-stack deployment—bringing models from Jupyter notebooks to production-ready web applications.
              </p>
            </motion.div>

            {/* Social Links */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.7 }}
              className="flex gap-4"
            >
              {socialLinks.map((link, index) => (
                <motion.a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.8 + index * 0.1 }}
                  whileHover={{ scale: 1.15, y: -3 }}
                  whileTap={{ scale: 0.95 }}
                  className="group relative"
                >
                  {/* Tooltip */}
                  <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-3 py-1 bg-dark-card border border-dark-border rounded-lg text-sm text-white opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                    {link.label}
                  </div>
                  
                  {/* Icon container */}
                  <div className="w-12 h-12 rounded-xl bg-dark-card border border-dark-border flex items-center justify-center transition-all duration-300 group-hover:border-primary/50 group-hover:shadow-lg group-hover:shadow-primary/20">
                    <link.icon 
                      size={22} 
                      className="text-gray-400 group-hover:text-white transition-colors" 
                    />
                  </div>
                  
                  {/* Ripple effect on hover */}
                  <div className="absolute inset-0 rounded-xl bg-primary/20 scale-0 group-hover:scale-100 transition-transform duration-300 -z-10" />
                </motion.a>
              ))}
            </motion.div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          100% {
            transform: translateX(200%);
          }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          background-size: 200% 200%;
          animation: gradient 8s ease infinite;
        }
      `}</style>
    </section>
  );
};

export default Developer;
