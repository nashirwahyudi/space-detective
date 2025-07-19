# ----- Stage 1: Build -----
FROM registry.cn-hangzhou.aliyuncs.com/libraries/node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm install

# Copy the rest of the source
COPY . .

# Build the app
RUN npm run build

# ----- Stage 2: Run -----
FROM registry.cn-hangzhou.aliyuncs.com/libraries/node:18-alpine

# Set working directory
WORKDIR /app

# Copy built app and dependencies
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Set environment variables (optional)
ENV NODE_ENV production
ENV PORT 3000

# Expose port
EXPOSE 3000

# Start the app
CMD ["npm", "start"]
