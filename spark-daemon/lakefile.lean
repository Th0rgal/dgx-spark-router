import Lake
open Lake DSL

package «spark-daemon» where
  version := v!"0.1.0"

require «lean-sys» from git
  "https://github.com/Th0rgal/lean-sys.git" @ "c8449e02ad65f7afa7c6b3a83e7174b35c8c109c"

@[default_target]
lean_lib «Spark» where
  roots := #[`Spark]
  precompileModules := false

@[default_target]
lean_exe «spark-daemon» where
  root := `Spark.Main
  supportInterpreter := false
