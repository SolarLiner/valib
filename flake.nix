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
        inherit (pkgs) lib stdenv;

        toolchain = (pkgs.rustChannelOf {
          toolchain = ./rust-toolchain;
          sha256 = "sha256-2Af13p12CWwmppsdujS1EeCQ41u0rMzJmqNh7WQ2QKM=";
        }).rust;
        naersk' = pkgs.callPackage naersk {
          cargo = toolchain;
          rustc = toolchain;
        };

        package' = { pname, display-name, gui ? false, vst3 ? false, clap ? false }:
          naersk'.buildPackage {
            name = pname;
            nativeBuildInputs = with pkgs; [ pkg-config python3 ];
            buildInputs = if gui then
              with pkgs;
              with pkgs.xorg; [
                jack2
                libGL
              ] ++ lib.optionals (!stdenv.isDarwin) [
                alsa-lib
                libX11
                libxcb
                libXcursor
                xcbutilwm
              ]
            else
              [ ];
            src = ./.;
            cargoBuildOptions = opts: [ "-p ${pname}" ] ++ opts;
            CREATE_VST3 = vst3;
            CREATE_CLAP = clap;
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
                if (( ''${CREATE_VST3} )); then
                  cp ''${LIBNAME} $out/lib/vst3/${display-name}.vst3
                fi
                if (( ''${CREATE_CLAP} )); then
                  cp ''${LIBNAME} $out/lib/clap/${display-name}.clap
                fi
              fi
            '';
          };
      in rec {
        packages.valib = naersk'.buildPackage {
          src = ./.;
          doDoc = true;
        };
        packages.abrasive = package' {
          pname = "abrasive";
          display-name = "Abrasive";
          gui = true;
          vst3 = true;
          clap = true;
        };
        defaultPackage = packages.valib;
        apps.abrasive = flake-utils.lib.mkApp { drv = packages.abrasive; };
      });
}
