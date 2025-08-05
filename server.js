const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5002;

// Middleware
app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'Python Scripts Service',
    timestamp: new Date().toISOString()
  });
});

// Execute script endpoint
app.post('/execute', (req, res) => {
  const { command } = req.body;
  
  if (!command) {
    return res.status(400).json({ success: false, error: 'Command is required' });
  }
  
  // Whitelist of allowed commands for security
  const allowedCommands = [
    'python props_enhanced.py',
    'python props_enhanced.py --tomorrow',
    'python teams_enhanced.py',
    'python teams_enhanced.py --tomorrow',
    'python insights_personalized_enhanced.py',
    'python daily_trends_generator.py',
    'python trendsnew.py'
  ];
  
  // Check if command is allowed
  if (!allowedCommands.some(cmd => command.startsWith(cmd))) {
    return res.status(403).json({ 
      success: false, 
      error: 'Command not allowed for security reasons'
    });
  }
  
  console.log(`Executing command: ${command}`);
  
  // Execute the command with a timeout
  const child = exec(command, { 
    maxBuffer: 1024 * 1024 * 10, // 10MB buffer
    timeout: 600000 // 10 minute timeout
  }, (error, stdout, stderr) => {
    if (error) {
      console.error(`Command execution error: ${error.message}`);
      return res.status(500).json({ 
        success: false, 
        error: error.message,
        stderr: stderr
      });
    }
    
    const output = stdout || stderr;
    console.log(`Command executed successfully: ${command}`);
    
    return res.json({
      success: true,
      output: output.substring(0, 2000), // Limit response size
      command,
      timestamp: new Date().toISOString()
    });
  });
  
  // Handle timeout
  setTimeout(() => {
    if (!child.killed) {
      child.kill();
      return res.status(408).json({
        success: false,
        error: 'Command execution timeout (10 minutes)'
      });
    }
  }, 600000);
});

app.listen(PORT, () => {
  console.log(`🚀 Python Scripts Service listening on port ${PORT}`);
  console.log(`📊 Ready to execute Python scripts`);
});