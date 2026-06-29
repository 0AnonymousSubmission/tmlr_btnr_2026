{
  description = "Bayesian Env";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
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
