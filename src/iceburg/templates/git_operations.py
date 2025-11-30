"""
Git Operations Integration

Provides:
- Git repository management
- Commit operations
- Branch management
- Status checking
- Remote operations
- Integration with IDE
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class GitOperationsGenerator:
    """Generates Git operations for IDE applications."""
    
    def __init__(self):
        self.git_commands = {
            'status': 'git status --porcelain',
            'add': 'git add',
            'commit': 'git commit',
            'push': 'git push',
            'pull': 'git pull',
            'branch': 'git branch',
            'checkout': 'git checkout',
            'merge': 'git merge',
            'log': 'git log --oneline',
            'diff': 'git diff'
        }
    
    def generate_git_operations_swift(self, features: List[str] = None) -> str:
        """
        Generate Swift code for Git operations.
        
        Args:
            features: List of Git features
            
        Returns:
            Swift code for Git operations
        """
        features = features or []
        
        swift_code = f'''import Foundation
import Process

// MARK: - Git Manager
class GitManager: ObservableObject {{
    @Published var repositoryPath: String?
    @Published var currentBranch: String?
    @Published var hasChanges = false
    @Published var stagedFiles: [String] = []
    @Published var unstagedFiles: [String] = []
    @Published var untrackedFiles: [String] = []
    @Published var commitHistory: [GitCommit] = []
    @Published var branches: [String] = []
    @Published var remotes: [GitRemote] = []
    
    private let gitCommands: [String: String] = {json.dumps(self.git_commands)}
    
    // MARK: - Repository Management
    
    func initializeRepository(at path: String) {{
        repositoryPath = path
        updateRepositoryStatus()
    }}
    
    func isGitRepository(at path: String) -> Bool {{
        let gitPath = URL(fileURLWithPath: path).appendingPathComponent(".git")
        return FileManager.default.fileExists(atPath: gitPath.path)
    }}
    
    func createRepository(at path: String) {{
        executeGitCommand("init", in: path) {{ success in
            if success {{
                self.repositoryPath = path
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    // MARK: - Status Operations
    
    func updateRepositoryStatus() {{
        guard let path = repositoryPath else {{ return }}
        
        // Get status
        executeGitCommand("status --porcelain", in: path) {{ output in
            DispatchQueue.main.async {{
                self.parseStatusOutput(output)
            }}
        }}
        
        // Get current branch
        executeGitCommand("branch --show-current", in: path) {{ output in
            DispatchQueue.main.async {{
                self.currentBranch = output.trimmingCharacters(in: .whitespacesAndNewlines)
            }}
        }}
        
        // Get branches
        executeGitCommand("branch -a", in: path) {{ output in
            DispatchQueue.main.async {{
                self.branches = self.parseBranches(output)
            }}
        }}
        
        // Get remotes
        executeGitCommand("remote -v", in: path) {{ output in
            DispatchQueue.main.async {{
                self.remotes = self.parseRemotes(output)
            }}
        }}
    }}
    
    private func parseStatusOutput(_ output: String) {{
        let lines = output.components(separatedBy: .newlines)
        var staged: [String] = []
        var unstaged: [String] = []
        var untracked: [String] = []
        
        for line in lines {{
            if line.isEmpty {{ continue }}
            
            let status = String(line.prefix(2))
            let filename = String(line.dropFirst(3))
            
            switch status {{
            case "M ", "A ", "D ":
                staged.append(filename)
            case " M", " D":
                unstaged.append(filename)
            case "??":
                untracked.append(filename)
            default:
                break
            }}
        }}
        
        stagedFiles = staged
        unstagedFiles = unstaged
        untrackedFiles = untracked
        hasChanges = !staged.isEmpty || !unstaged.isEmpty || !untracked.isEmpty
    }}
    
    private func parseBranches(_ output: String) -> [String] {{
        return output.components(separatedBy: .newlines)
            .filter {{ !$0.isEmpty }}
            .map {{ $0.replacingOccurrences(of: "* ", with: "").trimmingCharacters(in: .whitespacesAndNewlines) }}
    }}
    
    private func parseRemotes(_ output: String) -> [GitRemote] {{
        return output.components(separatedBy: .newlines)
            .filter {{ !$0.isEmpty }}
            .compactMap {{ GitRemote(from: $0) }}
    }}
    
    // MARK: - File Operations
    
    func addFile(_ file: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("add \\(file)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func addAllFiles() {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("add .", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func removeFile(_ file: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("rm \\(file)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func resetFile(_ file: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("reset HEAD \\(file)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    // MARK: - Commit Operations
    
    func commitChanges(message: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("commit -m \\"\\(message)\\"", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
                self.loadCommitHistory()
            }}
        }}
    }}
    
    func amendCommit(message: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("commit --amend -m \\"\\(message)\\"", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
                self.loadCommitHistory()
            }}
        }}
    }}
    
    // MARK: - Branch Operations
    
    func createBranch(_ branchName: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("checkout -b \\(branchName)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func switchBranch(_ branchName: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("checkout \\(branchName)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func deleteBranch(_ branchName: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("branch -d \\(branchName)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func mergeBranch(_ branchName: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("merge \\(branchName)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    // MARK: - Remote Operations
    
    func addRemote(name: String, url: String) {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("remote add \\(name) \\(url)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func pushToRemote(remote: String = "origin", branch: String? = nil) {{
        guard let path = repositoryPath else {{ return }}
        
        let branchName = branch ?? currentBranch ?? "main"
        executeGitCommand("push \\(remote) \\(branchName)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    func pullFromRemote(remote: String = "origin", branch: String? = nil) {{
        guard let path = repositoryPath else {{ return }}
        
        let branchName = branch ?? currentBranch ?? "main"
        executeGitCommand("pull \\(remote) \\(branchName)", in: path) {{ success in
            if success {{
                self.updateRepositoryStatus()
            }}
        }}
    }}
    
    // MARK: - History Operations
    
    func loadCommitHistory() {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("log --oneline -10", in: path) {{ output in
            DispatchQueue.main.async {{
                self.commitHistory = self.parseCommitHistory(output)
            }}
        }}
    }}
    
    private func parseCommitHistory(_ output: String) -> [GitCommit] {{
        return output.components(separatedBy: .newlines)
            .filter {{ !$0.isEmpty }}
            .compactMap {{ GitCommit(from: $0) }}
    }}
    
    // MARK: - Diff Operations
    
    func getDiff(for file: String? = nil) {{
        guard let path = repositoryPath else {{ return }}
        
        let command = file != nil ? "diff \\(file!)" : "diff"
        executeGitCommand(command, in: path) {{ output in
            // Handle diff output
        }}
    }}
    
    func getStagedDiff() {{
        guard let path = repositoryPath else {{ return }}
        
        executeGitCommand("diff --cached", in: path) {{ output in
            // Handle staged diff output
        }}
    }}
    
    // MARK: - Private Methods
    
    private func executeGitCommand(_ command: String, in path: String, completion: @escaping (String) -> Void) {{
        let process = Process()
        process.launchPath = "/usr/bin/git"
        process.arguments = command.components(separatedBy: " ")
        process.currentDirectoryPath = path
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        do {{
            try process.run()
            process.waitUntilExit()
            
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            
            completion(output)
        }} catch {{
            completion("")
        }}
    }}
}}

// MARK: - Git Data Structures

struct GitCommit {{
    let hash: String
    let message: String
    let author: String
    let date: Date
    
    init?(from line: String) {{
        let components = line.components(separatedBy: " ")
        guard components.count >= 2 else {{ return nil }}
        
        self.hash = components[0]
        self.message = components.dropFirst().joined(separator: " ")
        self.author = "Unknown" // Would need to parse from git log
        self.date = Date() // Would need to parse from git log
    }}
}}

struct GitRemote {{
    let name: String
    let url: String
    let type: String
    
    init?(from line: String) {{
        let components = line.components(separatedBy: .whitespaces)
        guard components.count >= 3 else {{ return nil }}
        
        self.name = components[0]
        self.url = components[1]
        self.type = components[2]
    }}
}}

// MARK: - Git UI Components
struct GitStatusView: View {{
    @ObservedObject var gitManager: GitManager
    
    var body: some View {{
        VStack(alignment: .leading, spacing: 8) {{
            HStack {{
                Image(systemName: "git.branch")
                Text(gitManager.currentBranch ?? "No branch")
                Spacer()
                if gitManager.hasChanges {{
                    Text("Modified")
                        .foregroundColor(.orange)
                        .font(.caption)
                }}
            }}
            
            if !gitManager.stagedFiles.isEmpty {{
                VStack(alignment: .leading) {{
                    Text("Staged Files:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    ForEach(gitManager.stagedFiles, id: \\.self) {{ file in
                        Text("• \\(file)")
                            .font(.caption)
                            .foregroundColor(.green)
                    }}
                }}
            }}
            
            if !gitManager.unstagedFiles.isEmpty {{
                VStack(alignment: .leading) {{
                    Text("Unstaged Files:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    ForEach(gitManager.unstagedFiles, id: \\.self) {{ file in
                        Text("• \\(file)")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }}
                }}
            }}
            
            if !gitManager.untrackedFiles.isEmpty {{
                VStack(alignment: .leading) {{
                    Text("Untracked Files:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    ForEach(gitManager.untrackedFiles, id: \\.self) {{ file in
                        Text("• \\(file)")
                            .font(.caption)
                            .foregroundColor(.red)
                    }}
                }}
            }}
        }}
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }}
}}'''
        
        return swift_code
    
    def generate_git_integration_swift(self, app_name: str, features: List[str] = None) -> str:
        """
        Generate complete Git integration Swift code.
        
        Args:
            app_name: Name of the application
            features: List of Git features
            
        Returns:
            Complete Swift code for Git integration
        """
        features = features or []
        
        # Generate the main Git operations code
        git_swift = self.generate_git_operations_swift(features)
        
        # Add feature-specific code
        if 'git-flow' in features:
            git_swift += self._generate_git_flow_integration()
        
        if 'git-hooks' in features:
            git_swift += self._generate_git_hooks_integration()
        
        if 'git-lfs' in features:
            git_swift += self._generate_git_lfs_integration()
        
        return git_swift
    
    def _generate_git_flow_integration(self) -> str:
        """Generate Git Flow integration code."""
        return '''

// MARK: - Git Flow Integration
extension GitManager {
    func startFeature(_ featureName: String) {
        createBranch("feature/\(featureName)")
    }
    
    func finishFeature(_ featureName: String) {
        switchBranch("develop")
        mergeBranch("feature/\(featureName)")
        deleteBranch("feature/\(featureName)")
    }
    
    func startRelease(_ releaseName: String) {
        createBranch("release/\(releaseName)")
    }
    
    func finishRelease(_ releaseName: String) {
        switchBranch("main")
        mergeBranch("release/\(releaseName)")
        switchBranch("develop")
        mergeBranch("release/\(releaseName)")
        deleteBranch("release/\(releaseName)")
    }
    
    func startHotfix(_ hotfixName: String) {
        createBranch("hotfix/\(hotfixName)")
    }
    
    func finishHotfix(_ hotfixName: String) {
        switchBranch("main")
        mergeBranch("hotfix/\(hotfixName)")
        switchBranch("develop")
        mergeBranch("hotfix/\(hotfixName)")
        deleteBranch("hotfix/\(hotfixName)")
    }
}'''
    
    def _generate_git_hooks_integration(self) -> str:
        """Generate Git hooks integration code."""
        return '''

// MARK: - Git Hooks Integration
extension GitManager {
    func installGitHooks() {
        guard let path = repositoryPath else { return }
        
        let hooksDir = URL(fileURLWithPath: path).appendingPathComponent(".git/hooks")
        
        // Install pre-commit hook
        let preCommitHook = """
#!/bin/sh
# Pre-commit hook
echo "Running pre-commit checks..."
# Add your pre-commit checks here
"""
        
        try? preCommitHook.write(to: hooksDir.appendingPathComponent("pre-commit"), atomically: true, encoding: .utf8)
        
        // Install commit-msg hook
        let commitMsgHook = """
#!/bin/sh
# Commit message hook
echo "Validating commit message..."
# Add your commit message validation here
"""
        
        try? commitMsgHook.write(to: hooksDir.appendingPathComponent("commit-msg"), atomically: true, encoding: .utf8)
    }
}'''
    
    def _generate_git_lfs_integration(self) -> str:
        """Generate Git LFS integration code."""
        return '''

// MARK: - Git LFS Integration
extension GitManager {
    func installGitLFS() {
        guard let path = repositoryPath else { return }
        
        executeGitCommand("lfs install", in: path) { success in
            if success {
            }
        }
    }
    
    func trackLargeFiles(_ patterns: [String]) {
        guard let path = repositoryPath else { return }
        
        for pattern in patterns {
            executeGitCommand("lfs track \"\(pattern)\"", in: path) { success in
                if success {
                }
            }
        }
    }
}'''
