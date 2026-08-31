{
  description = "Bayesian Env with Aim";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    # Define the custom Python package separately
    pkgs = import nixpkgs {
      inherit system;
    };

    python = pkgs.python313;
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.libxcb
        pkgs.libGL
        pkgs.glib
      ];
      packages = [
        python
        pkgs.uv
      ];

      shellHook = ''
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.libxcb pkgs.libGL pkgs.glib]}"
        export PROJECT_DIR=$PWD
        # --- PODMAN SERVICE ---
        DOCS_CONTAINER_NAME="docs-mcp-server-dev"
        DATA_DIR="$PWD/.docs-data"
        mkdir -p "$DATA_DIR"
        # Load GEMINI_API_KEY from sops-nix secrets
        GEMINI_API_KEY_FILE="$HOME/.config/sops-nix/secrets/geminiApi"
        if [ -f "$GEMINI_API_KEY_FILE" ]; then
          export GEMINI_API_KEY=$(cat "$GEMINI_API_KEY_FILE")
          echo "✓ Loaded GEMINI_API_KEY from sops-nix secrets"
        else
          echo "⚠️  Warning: GEMINI_API_KEY file not found at $GEMINI_API_KEY_FILE"
          echo "   Zen MCP server may not work properly."
        fi

        # cleanup() {
        #   echo "Stopping docs-mcp-server..."
        #   ${pkgs.podman}/bin/podman stop "$DOCS_CONTAINER_NAME" >/dev/null 2>&1 || true
        # }
        # trap cleanup EXIT

        # Start server if not running
        # if ! podman ps --format "{{.Names}}" | grep -q "^$DOCS_CONTAINER_NAME$"; then
        #   echo "🚀 Starting docs-mcp-server..."
        #   podman run --rm -d \
        #     --name "$DOCS_CONTAINER_NAME" \
        #     -v "$DATA_DIR:/data" \
        #     -p 6280:6280 \
        #     ghcr.io/arabold/docs-mcp-server:latest \
        #     --protocol http --host 0.0.0.0 --port 6280 --no-telemetry
        #   sleep 2
        # else
        #   echo "✓ docs-mcp-server is running."
        # fi
        # Get the Nix Python site-packages path
        export NIX_PYTHON_SITE_PACKAGES="${python}/${python.sitePackages}"

        # Create or Repair the UV venv symlinks
        if [ ! -d .venv ]; then
          echo "Creating UV virtual environment..."
          uv venv --python ${python}/bin/python
        elif ! .venv/bin/python --version >/dev/null 2>&1; then
          echo "🔗 Nix store path changed. Re-linking .venv interpreter..."
          uv venv --python ${python}/bin/python
        fi

        source .venv/bin/activate
        export PYTHONPATH="$NIX_PYTHON_SITE_PACKAGES:$PYTHONPATH"
        echo "✓ Environment ready!"
        echo "  Python: $(which python)"
        zsh
      '';
    };
  };
}
