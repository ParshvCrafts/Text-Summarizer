import { useEffect } from 'react';
import ParticleBackground from './components/ui/ParticleBackground';
import Navbar from './components/ui/Navbar';
import ScrollProgress from './components/ui/ScrollProgress';
import Hero from './components/sections/Hero';
import Summarizer from './components/sections/Summarizer';
import HowItWorks from './components/sections/HowItWorks';
import Features from './components/sections/Features';
import Examples from './components/sections/Examples';
import Performance from './components/sections/Performance';
import Developer from './components/sections/Developer';
import Footer from './components/sections/Footer';
import useLenis from './hooks/useLenis';

function App() {
  useLenis();

  useEffect(() => {
    console.log(
      '%c🧠 Text Summarizer',
      'font-size: 24px; font-weight: bold; color: #06b6d4;'
    );
    console.log(
      '%cBuilt by Parshv Patel | UC Berkeley',
      'font-size: 14px; color: #8b5cf6;'
    );
    console.log(
      '%c✨ Interested in how this was built? Check out the GitHub!',
      'font-size: 12px; color: #22d3ee;'
    );
    console.log('%chttps://github.com/ParshvCrafts', 'font-size: 12px; color: #9ca3af;');
  }, []);

  return (
    <div className="relative min-h-screen">
      <ScrollProgress />
      <ParticleBackground />
      <Navbar />
      <main className="relative z-10">
        <Hero />
        <Summarizer />
        <HowItWorks />
        <section id="features">
          <Features />
        </section>
        <section id="examples">
          <Examples />
        </section>
        <Performance />
        <Developer />
      </main>
      <Footer />
    </div>
  );
}

export default App;
