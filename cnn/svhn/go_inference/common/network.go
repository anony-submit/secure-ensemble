package common

import (
	"fmt"
	"net"
)

func FindAvailablePort() (int, error) {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return 0, fmt.Errorf("failed to resolve TCP addr: %v", err)
	}

	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return 0, fmt.Errorf("failed to listen on TCP addr: %v", err)
	}
	defer l.Close()

	return l.Addr().(*net.TCPAddr).Port, nil
}

func GetAvailableAddress() (string, error) {
	port, err := FindAvailablePort()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("localhost:%d", port), nil
}
