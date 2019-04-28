default: run

build:
	cargo build --release

clean:
	cargo clean

run: build
	time ./target/release/rust-tracer

build-fast:
	cargo build
