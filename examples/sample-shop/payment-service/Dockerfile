FROM golang:1.22 AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /payment-service .

FROM gcr.io/distroless/static
COPY --from=build /payment-service /payment-service
EXPOSE 9000
ENTRYPOINT ["/payment-service"]
