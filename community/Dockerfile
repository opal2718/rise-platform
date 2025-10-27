# Node.js 런타임 이미지 사용
FROM node:18-alpine

# 작업 디렉토리 설정
WORKDIR /usr/src/app

# package.json 및 package-lock.json 복사
# 캐싱을 위해 종속성 설치 전에 복사
COPY package*.json ./

# 종속성 설치
RUN npm install

# 애플리케이션 소스 코드 복사
COPY . .

# 애플리케이션이 수신할 포트 노출
EXPOSE 7000

# 애플리케이션 시작 명령어
CMD [ "npm", "start" ]