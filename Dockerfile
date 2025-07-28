# ----- Stage 1: Build -----
FROM alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/node:20.16 AS builder

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm install --legacy-peer-deps

# Copy the rest of the source
COPY . .

# Build the app
RUN npm run build

# ----- Stage 2: Run -----
FROM alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/node:20.16

# Switch to root user for package installation
USER root

# Install Python 3 and pip for chatbot functionality
RUN yum update -y && \
    yum install -y python3 python3-pip python3-devel gcc postgresql-devel && \
    yum clean all

# Set working directory
WORKDIR /app

# Copy built app and dependencies
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Copy chatbot folder and install Python dependencies
COPY chatbot ./chatbot
COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Set environment variables (optional)
ENV NODE_ENV production
ENV PORT 3000

# Expose port
EXPOSE 3000

# Create startup script
COPY <<EOF /app/start.sh
#!/bin/bash
# Start chatbot API in background
cd /app/chatbot/endpoint && python3 standalone_api.py --host 0.0.0.0 --port 8000 &
# Start Next.js app
cd /app && npm start
EOF

RUN chmod +x /app/start.sh

# Expose chatbot port
EXPOSE 8000

# Start both services
CMD ["/app/start.sh"]
