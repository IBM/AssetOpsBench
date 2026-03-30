# AObench

AssetOpsBench is the next generation agentic benchmark for industrial asset operations.

The proposed architecture iteration of the bench embraces the MCP and A2A protocols and includes new components to enable wider use of the bench.
The main aspects of the architecture iteration are:

1. A scenario server provides the input for each scenario to an agent and carries out the evaluation or grading of the agent’s response to the input.   This is currently work in progress, an implementation will follow in subsequent PRs.
2. An observability server leverages open telemetry tracing for insights on agent performance and provides an interactive dashboard. Several technologies are possible including mlflow and langfuse. Preliminary explorations are using mlflow.
3. A data layer including a lightweight enterprise asset management database and rest api, `eamlite`. This component stores scenario data in a similar way as production systems and will eventually enable MCP access.
4. Agents and tools should eventually be provided as A2A and MCP servers. This change should make it easier for developers of new agents to use agents already available on the bench.  
5. Servers comprising part of AssetOpsBench should be started with `podman compose up` or similar.

The figure below shows the main elements. The dev plan would be to introduce these elements one at a time.

<p align=center><img src="doc/arch_ao.png" width=50% height=50%></p>

## Directory tree

With the above architecture in mind, the structure of the  repository could evolve to look as shown below.
The top level `compose.yml` file spins up all the required services for the benchmark to function.

```sh
aobench
├── compose.yml
├── datalayer
│   └── eamlite
│       ├── src/... 
│       └── compose.yml
├── catalogue​
│   ├── agents​
│   │   ├── a2a_ci
│   │   │    ├── src/...  
│   │   │    └── containerfile
│   │   └── ...
│   └── tools
│       ├── mcp_iottool
│       │    ├── src/...
│       │    └── containerfile
│       └── ...
├── observability
│   └── mlflow
└── scenarioserver​
    ├── lib/...
    └── server
        └── containerfile
```
