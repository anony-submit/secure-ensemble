module cnn/cifar10/go_inference

go 1.22.0

require (
	github.com/anony-submit/snu-mghe v0.0.0
	github.com/ldsec/lattigo/v2 v2.3.0
	google.golang.org/grpc v1.70.0
	google.golang.org/protobuf v1.36.4
	secure-ensemble/pkg v0.0.0
)

require (
	golang.org/x/crypto v0.30.0 // indirect
	golang.org/x/net v0.32.0 // indirect
	golang.org/x/sys v0.28.0 // indirect
	golang.org/x/text v0.21.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20241202173237-19429a94021a // indirect
)

replace (
	github.com/anony-submit/snu-mghe => ../../../snu-mghe
	secure-ensemble/pkg => ../../../pkg
)
