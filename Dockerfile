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

# Install Python 3.9+ and pip for chatbot functionality
RUN yum update -y && \
    yum install -y python39 python39-pip python39-devel gcc postgresql-devel && \
    yum clean all && \
    alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.9 1

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

# Test all package availability first
COPY <<EOF /tmp/test_packages.py
import subprocess
import sys

packages = [
    "fastapi>=0.68.0,<0.84.0",
    "uvicorn[standard]>=0.15.0,<0.17.0",
    "python-multipart>=0.0.5",
    "asyncpg>=0.25.0,<0.27.0",
    "psycopg2-binary>=2.8.6",
    "sqlalchemy[asyncio]>=1.4.0",
    "sentence-transformers>=2.0.0",
    "transformers>=4.0.0,<4.19.0",
    "scikit-learn>=0.24.0,<0.25.0",
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    "joblib>=1.0.0",
    "shap>=0.40.0",
    "h3-py>=3.7.0",
    "folium>=0.12.0",
    "plotly>=5.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tiktoken>=0.3.0",
    "regex>=2022.0.0",
    "dashscope>=1.10.0",
    "pydantic>=1.8.0",
    "python-dotenv>=0.19.0",
    "aiofiles>=0.7.0",
    "pytest>=6.0.0",
    "pytest-asyncio>=0.18.0",
    "black>=22.0.0",
    "flake8>=4.0.0"
]

print("Testing package availability...")
for pkg in packages:
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', pkg, '-i', 'https://pypi.org/simple/'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ {pkg} - Available")
        else:
            print(f"✗ {pkg} - Not available")
            print(f"  Error: {result.stderr[:200]}")
    except Exception as e:
        # Try to get available versions
        base_pkg = pkg.split('>=')[0].split('==')[0].split('[')[0]
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', f'{base_pkg}==', '-i', 'https://pypi.org/simple/'], 
                                  capture_output=True, text=True, timeout=30)
            if "from versions:" in result.stderr:
                versions = result.stderr.split("from versions:")[1].split(")")[0].strip()
                print(f"✗ {pkg} - Available versions: {versions}")
            else:
                print(f"✗ {pkg} - Failed: {str(e)}")
        except:
            print(f"✗ {pkg} - Failed to check versions")

print("Package availability test completed!")
EOF

RUN python3 /tmp/test_packages.py

RUN pip3 install -r requirements.txt -i https://pypi.org/simple/

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
