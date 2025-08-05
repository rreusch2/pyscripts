FROM node:18

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-full \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create app directory
WORKDIR /app

# Install Node.js dependencies
COPY package*.json ./
RUN npm install

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt --break-system-packages

# Copy application code
COPY . .

# Expose the port
ENV PORT=5002
EXPOSE 5002

# Start the application
CMD ["node", "server.js"]