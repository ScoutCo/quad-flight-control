{
  description = "Development shell for flight-test scripts";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [ ps.pip ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.git
            pythonEnv
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              python -m venv .venv
              source .venv/bin/activate
              python -m pip install --upgrade pip
              python -m pip install -e ".[dev]"
            else
              source .venv/bin/activate
            fi
          '';
        };
      }
    );
}
