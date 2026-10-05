FROM oven/bun:1 AS base
# bun
WORKDIR /app

COPY package.json bun.lock ./

RUN bun install

COPY . .

CMD ["bun", "index.ts"]
