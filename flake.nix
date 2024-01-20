{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-mozilla = {
      url = "github:mozilla/nixpkgs-mozilla";
      flake = false;
    };
  };

  outputs = { self, flake-utils, naersk, nixpkgs, nixpkgs-mozilla }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = (import nixpkgs) {
          inherit system;
          overlays = [ (import nixpkgs-mozilla) ];
        };

        toolchain = (pkgs.rustChannelOf {
          toolchain = ./rust-toolchain;
          sha256 = "sha256-NNO9WVU8KfipdeTGgFnQ6Zlw3wImnN5RKAQPdAHG0d0=";
        }).rust;
        naersk' = pkgs.callPackage naersk {
          cargo = toolchain;
          rustc = toolchain;
        };

        package' = { pname, display-name, gui ? false }:
          naersk'.buildPackage {
            name = pname;
            nativeBuildInputs = with pkgs; [ pkg-config python3 ];
            buildInputs = if gui then
              with pkgs;
              with pkgs.xorg; [
                alsa-lib
                jack2
                libGL
                libX11
                libxcb
                libXcursor
                xcbutilwm
              ]
            else
              [ ];
            src = ./.;
            cargoBuildOptions = opts: [ "-p ${pname}" ] ++ opts;
            postInstall = ''
              OUT_DIR=''${PWD}/target/release
              LIBNAME=''${OUT_DIR}/lib${pname}.so
              EXENAME=''${OUT_DIR}/${pname}

              pwd
              ls -alh
              ls -alh $PWD/target
              ls -alh $OUT_DIR
              ls $LIBNAME $EXENAME

              if [[ -f $LIBNAME ]]; then
                mkdir -p $out/lib/{vst3,clap}
                cp ''${LIBNAME} $out/lib/vst3/${display-name}.vst3
                cp ''${LIBNAME} $out/lib/clap/${display-name}.clap
              fi
            '';
          };
      in rec {
        packages.workspace = naersk'.buildPackage {
          src = ./.;
          doDoc = true;
        };
        packages.abrasive = package' {
          pname = "abrasive";
          display-name = "Abrasive";
          gui = true;
        };
        defaultPackage = packages.workspace;
      });
}
