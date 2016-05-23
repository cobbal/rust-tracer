default: run

build:
	cargo build --release

run: build
	time ./target/release/rust-tracer

build-fast:
	cargo build
