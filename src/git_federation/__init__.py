"""
Git Federation

Manages git repositories across nodes in the federated network.
Each node hosts bare git repos locally and syncs with peers via QMP announcements.

Design principles:
- Local-first: every node has a full copy of repos it cares about
- No central server: repos replicate peer-to-peer
- QMP-based discovery: repos are announced over the mesh
- Standard git protocol: git:// or file:// — no custom protocol needed
"""

import json
import subprocess
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

QMP_MSG_REPOS_ANNOUNCE = "git.repos.announce"
QMP_MSG_REPOS_REQUEST = "git.repos.request"


@dataclass
class RepoManifest:
    """Metadata about a federated git repository."""
    name: str
    description: str
    owner_node_id: str
    created_at: float
    remotes: Dict[str, str] = field(default_factory=dict)  # node_id -> git_url

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'RepoManifest':
        return cls(**data)


class GitFederation:
    """
    Manages federated git repositories.

    Every node runs as both client and server:
    - It hosts bare repos in `repos_path`
    - It syncs with peers by fetching from their remotes
    - It announces its repos via QMP so peers can discover them
    """

    def __init__(self, repos_path: str, node_id: str, git_port: int = 9418):
        self.repos_path = Path(repos_path)
        self.repos_path.mkdir(parents=True, exist_ok=True)
        self.node_id = node_id
        self.git_port = git_port
        self.manifests: Dict[str, RepoManifest] = {}
        self._git_daemon_proc: Optional[subprocess.Popen] = None
        self._load_manifests()

    @property
    def _manifests_file(self) -> Path:
        return self.repos_path / "_manifests.json"

    def _load_manifests(self):
        """Load repo manifests from disk."""
        if self._manifests_file.exists():
            with open(self._manifests_file) as f:
                data = json.load(f)
            self.manifests = {k: RepoManifest.from_dict(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self.manifests)} repo manifests")

    def _save_manifests(self):
        """Persist repo manifests to disk."""
        with open(self._manifests_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.manifests.items()}, f, indent=2)

    def _run_git(self, *args, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        cmd = ["git"] + list(args)
        result = subprocess.run(
            cmd,
            cwd=str(cwd or self.repos_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def _local_url(self, repo_name: str) -> str:
        """Return the local file:// URL for a repo."""
        return str(self.repos_path / f"{repo_name}.git")

    def peer_url(self, host: str, repo_name: str) -> str:
        """Return the git:// URL for a repo on a peer node."""
        return f"git://{host}:{self.git_port}/{repo_name}.git"

    # -------------------------------------------------------------------------
    # Repo lifecycle
    # -------------------------------------------------------------------------

    def init_repo(self, name: str, description: str = "") -> Optional[RepoManifest]:
        """Initialize a new bare git repo on this node."""
        repo_path = self.repos_path / f"{name}.git"
        if repo_path.exists():
            logger.warning(f"Repo '{name}' already exists")
            return self.manifests.get(name)

        result = self._run_git("init", "--bare", str(repo_path))
        if result.returncode != 0:
            return None

        (repo_path / "description").write_text(description or name)

        manifest = RepoManifest(
            name=name,
            description=description,
            owner_node_id=self.node_id,
            created_at=time.time(),
            remotes={self.node_id: self._local_url(name)},
        )
        self.manifests[name] = manifest
        self._save_manifests()
        logger.info(f"Initialized repo: {name}")
        return manifest

    def list_repos(self) -> List[RepoManifest]:
        """Return manifests for all repos known to this node."""
        return list(self.manifests.values())

    def get_repo(self, name: str) -> Optional[RepoManifest]:
        """Get a repo manifest by name, or None if not found."""
        return self.manifests.get(name)

    def delete_repo(self, name: str) -> bool:
        """Remove a repo from this node (local data only)."""
        repo_path = self.repos_path / f"{name}.git"
        if not repo_path.exists():
            return False

        import shutil
        shutil.rmtree(repo_path)
        self.manifests.pop(name, None)
        self._save_manifests()
        logger.info(f"Deleted repo: {name}")
        return True

    # -------------------------------------------------------------------------
    # Peer sync
    # -------------------------------------------------------------------------

    def add_peer_remote(self, repo_name: str, peer_node_id: str, peer_host: str) -> bool:
        """Register a peer node as a git remote for a repo."""
        repo_path = self.repos_path / f"{repo_name}.git"
        if not repo_path.exists():
            logger.error(f"Repo '{repo_name}' not found locally")
            return False

        url = self.peer_url(peer_host, repo_name)
        remote_name = f"peer_{peer_node_id[:8]}"

        # Remove stale remote if present, then re-add
        self._run_git("remote", "remove", remote_name, cwd=repo_path)
        result = self._run_git("remote", "add", remote_name, url, cwd=repo_path)

        if result.returncode == 0:
            if repo_name in self.manifests:
                self.manifests[repo_name].remotes[peer_node_id] = url
                self._save_manifests()
            logger.info(f"Added remote '{remote_name}' -> {url} for '{repo_name}'")
            return True
        return False

    def sync_from_peer(self, repo_name: str, peer_node_id: str) -> bool:
        """Fetch latest changes from a peer's copy of a repo."""
        repo_path = self.repos_path / f"{repo_name}.git"
        if not repo_path.exists():
            logger.error(f"Repo '{repo_name}' not found locally; use clone_from_peer() first")
            return False

        remote_name = f"peer_{peer_node_id[:8]}"
        result = self._run_git("fetch", remote_name, "--prune", cwd=repo_path)
        if result.returncode == 0:
            logger.info(f"Synced '{repo_name}' from peer {peer_node_id[:16]}...")
            return True
        return False

    def clone_from_peer(
        self, repo_name: str, peer_host: str, peer_node_id: str
    ) -> Optional[RepoManifest]:
        """Clone a repo from a peer node to this node as a bare repo."""
        repo_path = self.repos_path / f"{repo_name}.git"

        if repo_path.exists():
            # Already have it — just sync
            logger.info(f"Repo '{repo_name}' already exists, syncing from peer")
            self.add_peer_remote(repo_name, peer_node_id, peer_host)
            self.sync_from_peer(repo_name, peer_node_id)
            return self.manifests.get(repo_name)

        url = self.peer_url(peer_host, repo_name)
        result = self._run_git("clone", "--bare", url, str(repo_path))
        if result.returncode != 0:
            return None

        manifest = RepoManifest(
            name=repo_name,
            description="",
            owner_node_id=peer_node_id,
            created_at=time.time(),
            remotes={
                peer_node_id: url,
                self.node_id: self._local_url(repo_name),
            },
        )
        self.manifests[repo_name] = manifest
        self._save_manifests()
        logger.info(f"Cloned '{repo_name}' from {peer_host}")
        return manifest

    def sync_all(self) -> Dict[str, bool]:
        """Fetch latest changes from all remotes for all local repos."""
        results = {}
        for manifest in self.manifests.values():
            for peer_node_id in manifest.remotes:
                if peer_node_id == self.node_id:
                    continue
                ok = self.sync_from_peer(manifest.name, peer_node_id)
                results[f"{manifest.name}:{peer_node_id[:8]}"] = ok
        return results

    # -------------------------------------------------------------------------
    # QMP integration
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Git daemon server
    # -------------------------------------------------------------------------

    def start_server(self) -> bool:
        """
        Start git daemon to serve repos to peers over git://.

        Only repos owned by this node get a git-daemon-export-ok file,
        so only those are served. Returns True if daemon started successfully.
        """
        if self._git_daemon_proc and self._git_daemon_proc.poll() is None:
            logger.debug("git daemon already running")
            return True

        # Mark owned repos as exportable
        for manifest in self.manifests.values():
            if manifest.owner_node_id == self.node_id:
                repo_path = self.repos_path / f"{manifest.name}.git"
                if repo_path.exists():
                    (repo_path / "git-daemon-export-ok").touch()

        cmd = [
            "git", "daemon",
            "--reuseaddr",
            f"--base-path={self.repos_path}",
            f"--port={self.git_port}",
            "--verbose",
            str(self.repos_path),
        ]

        try:
            self._git_daemon_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.info(f"git daemon started on port {self.git_port}")
            return True
        except FileNotFoundError:
            logger.error("git not found in PATH — cannot start git daemon")
            return False

    def stop_server(self):
        """Stop the git daemon subprocess."""
        if self._git_daemon_proc:
            self._git_daemon_proc.terminate()
            try:
                self._git_daemon_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._git_daemon_proc.kill()
            self._git_daemon_proc = None
            logger.info("git daemon stopped")

    def server_running(self) -> bool:
        """Return True if the git daemon is currently running."""
        return (
            self._git_daemon_proc is not None
            and self._git_daemon_proc.poll() is None
        )

    def get_announce_payload(self) -> dict:
        """Build a QMP payload announcing this node's repos."""
        return {
            "node_id": self.node_id,
            "git_port": self.git_port,
            "repos": [
                {"name": m.name, "description": m.description}
                for m in self.manifests.values()
                if m.owner_node_id == self.node_id
            ],
        }

    def handle_peer_announcement(self, payload: dict, peer_host: str):
        """
        Handle a QMP git.repos.announce message from a peer.

        Updates the local manifest with any new repos the peer hosts.
        Does NOT auto-clone; the operator decides what to clone.
        """
        peer_node_id = payload.get("node_id")
        if not peer_node_id:
            return

        repos = payload.get("repos", [])
        for repo_info in repos:
            repo_name = repo_info.get("name")
            if not repo_name:
                continue

            if repo_name in self.manifests:
                # Known repo: register the peer as a remote
                self.add_peer_remote(repo_name, peer_node_id, peer_host)
            else:
                # New repo: record for potential cloning
                manifest = RepoManifest(
                    name=repo_name,
                    description=repo_info.get("description", ""),
                    owner_node_id=peer_node_id,
                    created_at=time.time(),
                    remotes={peer_node_id: self.peer_url(peer_host, repo_name)},
                )
                self.manifests[repo_name] = manifest
                self._save_manifests()
                logger.info(
                    f"Discovered repo '{repo_name}' on peer {peer_node_id[:16]}..."
                )
