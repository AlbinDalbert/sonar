# Static build behind nginx, which proxies /tts-api to the Sonar API so the browser sees one
# origin. TTS_API_URL must resolve when the container starts.
#
#   docker build -t sonar-ui .
#   docker run -p 8080:8080 -e TTS_API_URL=http://host.docker.internal:8888 sonar-ui

# The output is plain files, so build on the native platform and only copy them per arch.
FROM --platform=$BUILDPLATFORM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm test && npm run build

# Runs as uid 101 and listens on 8080.
FROM nginxinc/nginx-unprivileged:1-alpine
ENV TTS_API_URL=http://sonar-api:8888
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
