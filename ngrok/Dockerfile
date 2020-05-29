FROM alpine as builder
ADD https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip /ngrok/bin/ngrok.zip
RUN apk add unzip
RUN unzip /ngrok/bin/ngrok.zip -d /ngrok/bin

FROM alpine
COPY --from=builder /ngrok/bin/ngrok /bin