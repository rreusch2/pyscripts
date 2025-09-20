const express = require('express');
const cors = require('cors');
const { exec, spawn } = require('child_process');
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

// Reusable whitelist check
const isAllowed = (command) => {
  const allowed = [
    'python props_enhanced.py',
    'python props_enhanced.py --tomorrow',
    'python teams_enhanced.py',
    'python teams_enhanced.py --tomorrow',
    'python insights_personalized_enhanced.py',
    'python daily_trends_generator.py',
    'python trendsnew.py'
  ];
  return allowed.some(prefix => command.startsWith(prefix));
};

// Execute script endpoint (non-streaming)
app.post('/execute', (req, res) => {
  const { command } = req.body;
  if (!command) {
    return res.status(400).json({ success: false, error: 'Command is required' });
  }
  if (!isAllowed(command)) {
    return res.status(403).json({ success: false, error: 'Command not allowed for security reasons' });
  }

  console.log(`Executing command: ${command}`);

  let responded = false;
  const timeoutMs = 600000; // 10 minutes

  const child = exec(command, { maxBuffer: 1024 * 1024 * 10, timeout: timeoutMs }, (error, stdout, stderr) => {
    if (responded) return;
    responded = true;
    clearTimeout(timeoutId);

    if (error) {
      console.error(`Command execution error: ${error.message}`);
      return res.status(500).json({ success: false, error: error.message, stderr });
    }
    const output = stdout || stderr;
    console.log(`Command executed successfully: ${command}`);
    return res.json({ success: true, output: String(output).substring(0, 2000), command, timestamp: new Date().toISOString() });
  });

  const timeoutId = setTimeout(() => {
    if (responded) return;
    try { child.kill(); } catch {}
    responded = true;
    return res.status(408).json({ success: false, error: 'Command execution timeout (10 minutes)' });
  }, timeoutMs);
});

// Live log streaming via Server-Sent Events (SSE)
// Usage: GET /execute/stream?command=<encoded command>
app.get('/execute/stream', (req, res) => {
  const command = req.query.command;
  if (!command || typeof command !== 'string') {
    return res.status(400).json({ success: false, error: 'Command is required' });
  }
  if (!isAllowed(command)) {
    return res.status(403).json({ success: false, error: 'Command not allowed for security reasons' });
  }

  // SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const send = (event, data) => {
    if (event) res.write(`event: ${event}\n`);
    res.write(`data: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`);
  };

  send('start', { command });

  // Use bash to interpret the full command string safely in a shell
  const child = spawn('bash', ['-lc', command], { env: process.env });

  child.stdout.on('data', (chunk) => {
    const text = chunk.toString();
    process.stdout.write(text); // mirror to service logs
    text.split(/\r?\n/).forEach(line => { if (line) send('stdout', line); });
  });

  child.stderr.on('data', (chunk) => {
    const text = chunk.toString();
    process.stderr.write(text);
    text.split(/\r?\n/).forEach(line => { if (line) send('stderr', line); });
  });

  child.on('close', (code) => {
    send('done', { code });
    res.end();
  });

  req.on('close', () => {
    try { child.kill(); } catch {}
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Python Scripts Service listening on port ${PORT}`);
  console.log(`📊 Ready to execute Python scripts`);
});