FROM oven/bun:1 AS base
# bun
WORKDIR /app

COPY package.json bun.lock ./

RUN bun install --frozen-lockfile --production

COPY . .

CMD ["bun", "index.ts"]
