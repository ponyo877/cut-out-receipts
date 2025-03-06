#!/bin/bash

# Go のWASMファイル wasm_exec.js をコピー
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .

# WASM用にコンパイル
GOOS=js GOARCH=wasm go build -o main.wasm

echo "WebAssemblyのビルドが完了しました。"