# -------------------------------------------------------------
# Stage 1: Build the React/Vite application
# -------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Cache dependencies layer
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source files and build production bundle
COPY . .
RUN npm run build

# -------------------------------------------------------------
# Stage 2: Serve with lightweight NGINX Alpine (~25MB)
# -------------------------------------------------------------
FROM nginx:alpine

# Copy built static files to NGINX default directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom NGINX config for SPA routing (fallback to index.html)
RUN printf 'server {\n\
    listen 80;\n\
    server_name localhost;\n\
\n\
    location / {\n\
        root /usr/share/nginx/html;\n\
        index index.html index.htm;\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    error_page 500 502 503 504 /50x.html;\n\
    location = /50x.html {\n\
        root /usr/share/nginx/html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
