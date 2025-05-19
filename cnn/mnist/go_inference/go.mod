module cnn/mnist/go_inference

go 1.22.0

require (
	github.com/anony-submit/snu-mghe v0.0.0
	github.com/ldsec/lattigo/v2 v2.3.0
	google.golang.org/grpc v1.69.4
	google.golang.org/protobuf v1.35.1
	secure-ensemble/pkg v0.0.0
)

require (
	golang.org/x/crypto v0.28.0 // indirect
	golang.org/x/net v0.30.0 // indirect
	golang.org/x/sys v0.26.0 // indirect
	golang.org/x/text v0.19.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20241015192408-796eee8c2d53 // indirect
)

replace (
	github.com/anony-submit/snu-mghe => ../../../snu-mghe
	secure-ensemble/pkg => ../../../pkg
)
