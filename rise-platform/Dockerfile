# Stage 1: build
FROM node:18 AS builder

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy all source files
COPY . .

# Build the production version of Next.js
RUN npm run build

# Stage 2: production
FROM node:18-alpine

WORKDIR /app

# Copy only necessary files from the builder
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Optional: install only production dependencies (if not copied from builder)
# RUN npm install --production

# Expose port 3000 for GCP
EXPOSE 3000

# Run Next.js in production mode and bind to all interfaces
ENV PORT=3000
ENV HOST=0.0.0.0
CMD ["npm", "run", "start"]
