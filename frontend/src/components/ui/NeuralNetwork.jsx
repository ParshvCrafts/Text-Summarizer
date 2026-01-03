import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const NeuralNetwork = ({ isProcessing = false }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let time = 0;

    const nodes = [
      // Input layer
      { x: 50, y: 30, layer: 0 },
      { x: 50, y: 50, layer: 0 },
      { x: 50, y: 70, layer: 0 },
      // Hidden layer 1
      { x: 150, y: 20, layer: 1 },
      { x: 150, y: 40, layer: 1 },
      { x: 150, y: 60, layer: 1 },
      { x: 150, y: 80, layer: 1 },
      // Hidden layer 2
      { x: 250, y: 30, layer: 2 },
      { x: 250, y: 50, layer: 2 },
      { x: 250, y: 70, layer: 2 },
      // Output layer
      { x: 350, y: 50, layer: 3 },
    ];

    const connections = [];
    nodes.forEach((node, i) => {
      nodes.forEach((otherNode, j) => {
        if (otherNode.layer === node.layer + 1) {
          connections.push({ from: i, to: j });
        }
      });
    });

    const draw = () => {
      ctx.clearRect(0, 0, 400, 100);
      time += 0.02;

      // Draw connections
      connections.forEach((conn, index) => {
        const from = nodes[conn.from];
        const to = nodes[conn.to];
        
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        
        if (isProcessing) {
          const pulse = Math.sin(time * 3 + index * 0.5) * 0.5 + 0.5;
          ctx.strokeStyle = `rgba(6, 182, 212, ${0.1 + pulse * 0.4})`;
          ctx.lineWidth = 1 + pulse;
        } else {
          ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      });

      // Draw nodes
      nodes.forEach((node, index) => {
        ctx.beginPath();
        
        let radius = 6;
        let opacity = 0.6;
        
        if (isProcessing) {
          const pulse = Math.sin(time * 4 + index * 0.3) * 0.5 + 0.5;
          radius = 6 + pulse * 3;
          opacity = 0.6 + pulse * 0.4;
        }
        
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6, 182, 212, ${opacity})`;
        ctx.fill();
        
        // Glow effect
        if (isProcessing) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(6, 182, 212, 0.2)`;
          ctx.fill();
        }
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => cancelAnimationFrame(animationFrameId);
  }, [isProcessing]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex items-center justify-center"
    >
      <canvas
        ref={canvasRef}
        width={400}
        height={100}
        className="opacity-80"
      />
    </motion.div>
  );
};

export default NeuralNetwork;
