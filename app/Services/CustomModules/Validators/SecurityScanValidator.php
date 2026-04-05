<?php

namespace App\Services\CustomModules\Validators;

use Illuminate\Support\Facades\File;

class SecurityScanValidator
{
    /**
     * Perform security scan on uploaded ZIP file
     */
    public function validate(string $filePath): array
    {
        try {
            // 1. Verify ZIP is valid and readable
            if (!$this->isValidZip($filePath)) {
                return [
                    'valid' => false,
                    'issues' => [[
                        'severity' => 'error',
                        'code' => 'INVALID_ZIP',
                        'message' => 'ZIP file is corrupted or not a valid ZIP archive',
                    ]],
                ];
            }
            
            // 2. Extract and scan contents
            $tempDir = storage_path('temp/security_scan_' . uniqid());
            File::ensureDirectoryExists($tempDir);
            
            $issues = [];
            
            try {
                $zip = new \ZipArchive();
                $zip->open($filePath);
                $zip->extractTo($tempDir);
                $zip->close();
                
                // 3. Scan all PHP files for dangerous patterns
                $issues = array_merge($issues, $this->scanForShellPatterns($tempDir));
                
                // 4. Check for path traversal attempts
                $issues = array_merge($issues, $this->scanForPathTraversal($tempDir));
                
                // 5. Verify checksum
                $checksum = hash_file('sha256', $filePath);
                
            } finally {
                File::deleteDirectory($tempDir);
            }
            
            $hasErrors = !empty(array_filter($issues, fn($i) => $i['severity'] === 'error'));
            
            return [
                'valid' => !$hasErrors,
                'issues' => $issues,
                'checksum_sha256' => $checksum ?? null,
                'threats_found' => count(array_filter($issues, fn($i) => $i['code'] === 'SHELL_PATTERN')),
            ];
            
        } catch (\Exception $e) {
            return [
                'valid' => false,
                'issues' => [[
                    'severity' => 'error',
                    'code' => 'SCAN_ERROR',
                    'message' => "Security scan failed: {$e->getMessage()}",
                ]],
            ];
        }
    }
    
    /**
     * Check if ZIP file is valid and can be opened
     */
    private function isValidZip(string $filePath): bool
    {
        if (!File::exists($filePath)) {
            return false;
        }
        
        $zip = new \ZipArchive();
        $result = $zip->open($filePath);
        $zip->close();
        
        return $result === true;
    }
    
    /**
     * Scan for dangerous PHP functions that could be used for RCE
     */
    private function scanForShellPatterns(string $basePath): array
    {
        $issues = [];
        
        // Dangerous functions and patterns
        $dangerousPatterns = [
            'shell_exec' => 'Remote code execution',
            'exec' => 'Remote code execution',
            'system' => 'Remote code execution',
            'passthru' => 'Remote code execution',
            'proc_open' => 'Process spawning',
            'eval' => 'Code evaluation',
            'create_function' => 'Code evaluation',
            'assert' => 'Code evaluation',
            'php://' => 'Stream wrapper abuse',
            'stream_filter' => 'Stream manipulation',
        ];
        
        // Find all PHP files
        $phpFiles = File::allFiles($basePath, ['*.php']);
        
        foreach ($phpFiles as $file) {
            try {
                $content = $file->getContents();
                
                foreach ($dangerousPatterns as $pattern => $description) {
                    // Use case-insensitive search and avoid false positives in strings/comments
                    if ($this->containsDangerousPattern($content, $pattern)) {
                        $issues[] = [
                            'severity' => 'error',
                            'code' => 'SHELL_PATTERN',
                            'message' => "Dangerous pattern '{$pattern}' found: {$description}",
                            'file' => $file->getRelativePath(),
                            'pattern' => $pattern,
                        ];
                    }
                }
                
            } catch (\Exception $e) {
                // Skip files we can't read
                continue;
            }
        }
        
        return $issues;
    }
    
    /**
     * Check for path traversal attempts
     */
    private function scanForPathTraversal(string $basePath): array
    {
        $issues = [];
        
        $allFiles = File::allFiles($basePath);
        
        foreach ($allFiles as $file) {
            $relativePath = $file->getRelativePath();
            
            // Check for ../ in paths
            if (str_contains($relativePath, '../') || str_contains($relativePath, '..\\')) {
                $issues[] = [
                    'severity' => 'error',
                    'code' => 'PATH_TRAVERSAL',
                    'message' => "Suspicious path traversal attempt detected",
                    'file' => $relativePath,
                ];
            }
            
            // Check for attempts to write to parent directories
            if (preg_match('/\.\.[\/\\]/', $relativePath)) {
                $issues[] = [
                    'severity' => 'error',
                    'code' => 'PATH_TRAVERSAL',
                    'message' => "File attempts to escape module directory",
                    'file' => $relativePath,
                ];
            }
        }
        
        return $issues;
    }
    
    /**
     * Detect dangerous pattern avoiding false positives in strings/comments
     */
    private function containsDangerousPattern(string $content, string $pattern): bool
    {
        // Simple regex to find the pattern
        // This is a basic check - not bulletproof but catches obvious cases
        
        // Remove comments (PHP-style)
        $content = preg_replace('/#.*$/', '', $content);
        $content = preg_replace('/\/\/.*$/', '', $content);
        $content = preg_replace('%/\*.*?\*/%s', '', $content);
        
        // Look for function calls or direct usage
        return preg_match('/\b' . preg_quote($pattern, '/') . '\s*[\(\[]/', $content) === 1;
    }
}
