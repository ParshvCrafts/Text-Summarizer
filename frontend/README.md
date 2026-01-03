# Text Summarizer - Frontend

A stunning, portfolio-worthy React frontend for the AI-powered Text Summarizer.

## Features

- **Neural Network Visualization** - Animated nodes showing processing state
- **Typewriter Effect** - Summary appears character by character
- **Glassmorphism UI** - Modern, sleek card designs
- **Particle Background** - Subtle AI-themed ambient motion
- **Confidence Meter** - Animated circular gauge
- **Dark Mode** - Professional, eye-catching design
- **Responsive Design** - Works on all devices

## Tech Stack

- **React 19** - Latest React with hooks
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling (via CDN)
- **Framer Motion** - Smooth animations
- **Lucide React** - Beautiful icons

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # Reusable UI components
│   │   │   ├── ParticleBackground.jsx
│   │   │   ├── ConfidenceMeter.jsx
│   │   │   ├── NeuralNetwork.jsx
│   │   │   └── Navbar.jsx
│   │   └── sections/        # Page sections
│   │       ├── Hero.jsx
│   │       ├── Summarizer.jsx
│   │       ├── HowItWorks.jsx
│   │       ├── Features.jsx
│   │       ├── Examples.jsx
│   │       ├── Performance.jsx
│   │       └── Footer.jsx
│   ├── hooks/
│   │   └── useTypewriter.js
│   ├── utils/
│   │   └── api.js
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── index.html
├── vite.config.js
└── package.json
```

## API Configuration

The frontend connects to the backend API via proxy. Configure in `vite.config.js`:

```js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

For production, set the `VITE_API_URL` environment variable.

## Animations Implemented

1. **Particle Network Background** - Interactive particles with mouse tracking
2. **Hero Text Reveal** - Staggered fade-in animations
3. **Scroll Animations** - Elements animate on viewport entry
4. **Neural Network Processing** - Pulsing nodes during inference
5. **Typewriter Effect** - Character-by-character summary reveal
6. **Confidence Meter** - Animated circular progress
7. **Micro-interactions** - Button hovers, card lifts, tooltips
8. **Example Carousel** - Smooth slide transitions

## Build Output

```
dist/
├── index.html        (~1.8 KB)
├── assets/
│   ├── index.css     (~3 KB)
│   └── index.js      (~345 KB, ~110 KB gzipped)
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
