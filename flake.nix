{
  description = "Nix flake for running scripts/exec_path_follow.py";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      perSystem =
        f:
        forAllSystems (
          system:
          let
            pkgs = import nixpkgs { inherit system; };
            pythonEnv = pkgs.python312.withPackages (
              ps: with ps; [
                numpy
                pymavlink
              ]
            );
            execPathFollowBin = pkgs.writeShellScriptBin "exec-path-follow" ''
              exec ${pythonEnv}/bin/python ${./scripts/exec_path_follow.py} "$@"
            '';
          in
          f { inherit pkgs pythonEnv execPathFollowBin; }
        );
    in
    {
      devShells = perSystem (
        { pkgs, pythonEnv, execPathFollowBin }:
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            shellHook = ''
              export PATH=$PATH:${execPathFollowBin}/bin
              echo "Loaded devshell with python environment."
            '';
          };
        }
      );

      apps = perSystem (
        { execPathFollowBin }:
        {
          exec-path-follow = {
            type = "app";
            program = "${execPathFollowBin}/bin/exec-path-follow";
          };
        }
      );
    };
}
