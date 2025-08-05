# Python Scripts Service

This service executes Python scripts for the Parley App admin panel.

## Overview

The Python Scripts Service provides a secure HTTP interface to execute specific Python scripts remotely. It's designed to be called by the main backend when admin commands are triggered.

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /execute` - Execute a Python script

### Execute Script

```bash
POST /execute
Content-Type: application/json

{
  "command": "python props_enhanced.py"
}
```

**Allowed Commands:**
- `python props_enhanced.py`
- `python props_enhanced.py --tomorrow`
- `python teams_enhanced.py`
- `python teams_enhanced.py --tomorrow`
- `python insights_personalized_enhanced.py`
- `python daily_trends_generator.py`

## Environment Variables

Set these in your deployment:
- `PORT` - Port to run the service (default: 5002)
- `STATMUSE_API_URL` - URL of the StatMuse API service
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `OPENAI_API_KEY` - OpenAI API key
- Any other environment variables your scripts need

## Security

- Only whitelisted commands are allowed
- Commands have a 5-minute timeout
- Output is limited to prevent memory issues