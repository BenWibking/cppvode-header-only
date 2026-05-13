#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s INPUT [OUTPUT]\n' "$(basename "$0")" >&2
    printf 'Strip quoted local header includes from a concatenated single-file translation unit.\n' >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

input=$1
output=${2:-$input}

if [[ ! -f "$input" ]]; then
    printf 'error: input file not found: %s\n' "$input" >&2
    exit 1
fi

tmp=$(mktemp "${output}.tmp.XXXXXX")
cleanup() {
    rm -f "$tmp"
}
trap cleanup EXIT

awk '
    /^[[:space:]]*#[[:space:]]*include[[:space:]]*"[^"]+\.(h|hh|hpp|hxx)"[[:space:]]*$/ {
        next
    }
    {
        print
    }
' "$input" > "$tmp"

if [[ "$output" == "$input" ]]; then
    mv "$tmp" "$input"
else
    mv "$tmp" "$output"
fi

trap - EXIT
