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

# Install Python 3 and pip for chatbot functionality
RUN yum update -y && \
    yum install -y python3 python3-pip && \
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
RUN pip3 install -r requirements.txt

# Set environment variables (optional)
ENV NODE_ENV production
ENV PORT 3000

# Expose port
EXPOSE 3000

# Start the app
CMD ["npm", "start"]
