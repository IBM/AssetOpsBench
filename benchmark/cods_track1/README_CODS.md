# AssetOpsBench Setup Guide

This guide describes how to fork, configure, and run the **AssetOpsBench** project using Conda and Docker on macOS.

---

## 1. Fork to Your Own Project

This guide shows how to create your own copy of the **IBM/AssetOpsBench** project on GitHub, using only the web browser.  
A fork lets you experiment and make changes without affecting the original project.

---

### Step 1 — Log in to GitHub

1. Open [https://github.com](https://github.com) in your browser.
2. Sign in with your GitHub account.
   - If you don’t have one, click **Sign up** to create a free account.

---

### Step 2 — Navigate to the Original Repository

1. In the search bar at the top of GitHub, type **IBM/AssetOpsBench**.
2. Click on the repository link, or go directly to:  
   👉 [https://github.com/IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench)

---

### Step 3 — Required Branch for This Work

The repository contains multiple branches, but for this purpose you will only use:

- **`main`** → this branch contains the competition-specific code and materials.

👉 You still need to fork with **all branches** available so you can access `main`.

---

### Step 4 — Create the Fork

1. On the top-right of the page, click the **Fork** button.

2. On the fork setup page:

   - Make sure your own GitHub account is selected as the destination.
   - **Important:** Uncheck the option **“Copy the `main` branch only.”**
     - Leaving it checked will copy only `main`.
     - Unchecking ensures you also get **`main`** and other branches.

3. Click **Create fork**.

---

### Step 5 — Switch to the `main` Branch

1. Go to your forked repository:  
   👉 `https://github.com/<your-username>/AssetOpsBench`

2. Open the **Branch dropdown menu** (upper-left, just above the file list).
3. Select **`main`** from the list.
   - This switches your forked repository view to the competition branch.

---

### Step 6 — Verify Your Fork

- After switching, confirm that the branch name `main` is shown at the top-left of the repository.
- All further work for the competition should be done in this branch.

---

### Important Notes

- Always uncheck **“Copy the `main` branch only”** when forking. Otherwise you won’t see `main`.
- Even though other branches are copied, you only need to use **`main`** for this competition.

## 2. Clone to Your Local Machine (Using GitHub Desktop)

Once you have forked the repository on GitHub, you may want to work with it locally on your computer.  
Here we use **GitHub Desktop** to illustrate the process.

---

### Step 1 — Install GitHub Desktop

1. Download GitHub Desktop from: https://desktop.github.com
2. Install and open the application.
3. Sign in with your GitHub account.

---

### Step 2 — Locate Your Fork

1. In your browser, go to your forked repository:  
   `https://github.com/<your-username>/AssetOpsBench`
2. Click the green **Code** button.
3. Select **Open with GitHub Desktop**.
   - This will launch GitHub Desktop and prepare to clone your fork.

---

### Step 3 — Choose the Local Path

1. In GitHub Desktop, choose where to save the project locally.
2. Set the path to:
   local path: <your_local_directory>/codabench/AssetOpsBench

### Step 4 — Switch to the `main` Branch

⚠️ For this competition, you must work **only** on the `main` branch.

1. In GitHub Desktop, use the **Current Branch** dropdown (top bar).
2. Select `main` from the list.
   - If you don’t see it, ensure you unchecked “Copy the main branch only” when you forked.

---

### Step 5 — Verify

1. The branch name shown at the top of GitHub Desktop should be **main**.
2. Confirm the file view matches the content of this branch.

---

✅ You have successfully cloned your fork locally with GitHub Desktop and switched to the **`main`** branch. All further work should be done in this branch.

## 3. Docker Setup (Vendor‑Agnostic — Rancher Desktop as Example)

You can use **any Docker‑compatible engine**. Popular choices include:

- **Docker Desktop** (macOS/Windows) — most common, easy setup.
- **Rancher Desktop** (macOS/Windows/Linux) — open‑source; great Apple Silicon support. **We’ll use this as the example.**
- **Colima** (macOS) — lightweight, Homebrew‑friendly.
- **Podman / Podman Desktop** (Linux/macOS/Windows) — Docker‑compatible CLI via `podman`.

> **Tip:** Run **only one** engine at a time. If multiple are installed, whichever provides the `docker` CLI on your `PATH` will “win.”

### 3.1 Choose, Install & (If Using Rancher Desktop) Configure

1. **Pick one engine** above and install it.
2. **If using Rancher Desktop** → **Preferences** → **Virtual Machine → Emulation**:
   - **Virtual Machine Type**
     - `VZ` (recommended on Apple Silicon) → uses Apple’s Virtualization framework.
     - `QEMU` → software emulator (slower; use only if VZ is unavailable).
   - **VZ Options**: enable **Rosetta support** to run `amd64` (x86) Docker images on Apple Silicon.
3. Click **Apply**. Rancher Desktop restarts with the new settings.

### 3.2 Verify the Docker CLI

Confirm that the `docker` command points to your chosen engine and the daemon is running:

`HINT` : For macos, navigate to official docker website, download the `.dmg` file corresponding to intel/silicon architecture, after installation, can be started up using spotlight search.

```bash
docker version
docker info
docker context ls
which docker
```

**Expected path (example for Rancher Desktop):**

```
~/.rd/bin/docker
```

- On macOS this expands to `/Users/<your-username>/.rd/bin/docker`.
- On Linux it often expands to `/home/<your-username>/.rd/bin/docker` (example).

If you see `/usr/local/bin/docker`, another runtime (e.g., Docker Desktop) is active. Quit it and ensure your chosen engine is running.  
If needed, switch contexts (examples):

```bash
docker context use rancher-desktop   # Rancher Desktop
docker context use default           # Docker Desktop
docker context use colima            # Colima
```

### 3.3 (If Kubernetes is On) Ensure Certificates Are Fresh

If you enable Kubernetes in Rancher Desktop, its certificates can expire and block startup.

**Quick checks:**

```bash
kubectl version
kubectl get nodes
kubectl cluster-info
```

**If errors mention expired or invalid certificates:**

- In Rancher Desktop: **Preferences → Kubernetes → Reset Kubernetes** (regenerates certificates).
- If problems persist: **Preferences → Troubleshooting → Factory Reset** (removes images/containers and resets certificates).

---

## 4. Environment Variables

You can use [.env](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/cods_track1/.env) or add your environment variables to `.env` which can be passed to containers through `docker-compose.yml`.

Fill in your `.env.local` file with the following (Please see Notes, you do not need all the value to be filled.):

```
cd AssetOpsBench/benchmark/cods_track1
cp .env .env.local
```

This will copy the content from .env to .env.local file and will not be tracked by git version control, ensuring your API keys remain protected.

```env
HF_APIKEY= <<Hugging face API key>>
WATSONX_APIKEY=
WATSONX_PROJECT_ID=
WATSONX_URL=
```

### Notes

- For the **credential related to WATSONX**, please follow this [link](https://www.codabench.org/forums/10049/1449/) to gain the WATSONX.AI account request
- **Hugging Face API key**: Required to fetch utterances for this exercise.  
  You can create one from your Hugging Face account [here](https://huggingface.co/join).
- **Skyspark credentials**: Required but cal leave empty
- **OpenAI API key**: Required but should leave as empty

---

## 5. Build and Run

### Pre-built Docker Images

We provide two base Docker images `quay.io/assetopsbench/assetopsbench-basic` and `quay.io/assetopsbench/assetopsbench-extra` which serves as the AssetOpsBench environment. The base images provide

- `quay.io/assetopsbench/assetopsbench-basic`: A minimal Python 3.12 environment with only essential Python packages installed. Suitable for lightweight benchmarking tasks or custom setup.
- `quay.io/assetopsbench/assetopsbench-extra`: A full-featured Python 3.12 environment with all components pre-installed, including Individual agents (e.g., IoT Agent), MetaAgent, and AgentHive.

The python environment in both docker images use Conda to manage. The name of pre-built Conda environment is `assetopsbench`. For full list of installed packages, please refer to [basic_requirements.txt](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/basic_requirements.txt) and [extra_requirements.txt](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/extra_requirements.txt).

### Run Benchmark Script

In [docker-compose.yml](https://github.com/IBM/AssetOpsBench/blob/Competition_CODS/benchmark/cods_track1/docker-compose.yml), we mount the benchmark scripts ([entrypoint.sh](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/cods_track1/entrypoint.sh) and [run_track_1.py](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/cods_track1/run_track_1.py) and scenario files as volumes to assetopsbench container.

Now we run the following command,

```commandline
cd /path/to/AssetOpsBench
chmod +x benchmark/cods_track1/entrypoint.sh
docker-compose -f benchmark/cods_track1/docker-compose.yml up
```

Note that in `entrypoint.sh`, we activate the python environment by `conda activate assetopsbench` first.

## 6. Useful Links

### Project

- [AssetOpsBench GitHub](https://github.com/IBM/AssetOpsBench) — Source code and issues.
- [Benchmark README](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/README.md) — How to build and run the benchmark stack.

### Container Engines (pick one)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — Official Docker engine + UI for macOS/Windows; simplest “it just works” option.
- [Rancher Desktop](https://rancherdesktop.io/) — Open-source engine with Moby/containerd; good Apple Silicon support; optional built-in Kubernetes (k3s).
- [Colima](https://github.com/abiosoft/colima) — Lightweight Docker-compatible runtime for macOS (via Lima); great for Homebrew users.
- [Podman Desktop](https://podman-desktop.io/) — Docker-compatible CLI/UX with a daemonless engine; works on Linux/macOS/Windows.

> Tip: Run **only one** engine at a time. If multiple are installed, whichever provides the `docker` CLI on your `PATH` will “win.”

### (Optional) Local Kubernetes Runtimes

If you want a local Kubernetes cluster (for `kubectl`), any of the below work:

- [kind](https://kind.sigs.k8s.io/) — Runs Kubernetes “in Docker”; great for CI and quick clusters.
- [minikube](https://minikube.sigs.k8s.io/) — Popular single-node local Kubernetes with many drivers.
- [k3d](https://k3d.io/) — Runs lightweight k3s in Docker; very fast to start.

### Image Registries

- [Docker Hub](https://hub.docker.com/) — Default public image registry used by `docker pull`.
- [Quay.io](https://quay.io/) — Red Hat/Quay container registry; often used for OSS and enterprise images.
- [GitHub Container Registry (GHCR)](https://github.com/features/packages) — Stores images in GitHub under `ghcr.io/<owner>/<image>`.
- [Amazon ECR](https://aws.amazon.com/ecr/) — Private registry integrated with AWS (IAM/permissions, ECS/EKS).
- [Google Artifact Registry](https://cloud.google.com/artifact-registry) — GCP registry for Docker images and more (replaces GCR).
- [Azure Container Registry](https://azure.microsoft.com/products/container-registry/) — Private registry integrated with Azure (AKS).
