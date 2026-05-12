.PHONY: all clean

all:
	TMPDIR=/private/tmp cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	TMPDIR=/private/tmp cmake --build build -j

clean:
	cmake -E rm -rf build
