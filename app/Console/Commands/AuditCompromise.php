<?php

namespace App\Console\Commands;

use FilesystemIterator;
use Illuminate\Console\Command;
use RecursiveDirectoryIterator;
use RecursiveIteratorIterator;
use SplFileInfo;
use Throwable;

class AuditCompromise extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'security:audit-compromise
        {--path=* : Web root or upload directory to scan recursively}
        {--log=* : Access or error log to inspect}
        {--max-bytes=52428800 : Maximum file size to inspect for content signatures}';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Read-only scan for indicators associated with the MagicAI upload RCE';

    /** @var array<int, string> */
    private const MALWARE_HASHES = [
        '4b447685f12e2cda63421298293fd8eaafbe67e52b9d2816d088ab2b52fbf113',
        '8bfbc4422da0938df0a0c1a0b6acb6cd1f21029a4808cde8841ba4b908b5cad7',
        '319eea349bfec23997d1e9c4a9e76deedf7e2fff4c581eec979f07173af6417a',
    ];

    /** @var array<int, string> */
    private const SHIPPED_SECURITY_FILE_HASHES = [
        '9cdfe879f4664135378e02b5a5f2ebdd4b54efca7a35c20a7e225884fca2891e',
    ];

    /** @var array<int, string> */
    private const SUSPICIOUS_EXTENSIONS = [
        'php', 'php3', 'php4', 'php5', 'php7', 'php8', 'phtml', 'pht', 'phar', 'phps',
        'cgi', 'shtml', 'asp', 'aspx', 'jsp', 'htaccess', 'user.ini',
    ];

    /**
     * Execute the console command.
     */
    public function handle(): int
    {
        $findings = [];
        $errors = [];
        $maxBytes = max(1024, (int) $this->option('max-bytes'));

        foreach ((array) $this->option('path') as $path) {
            $this->scanPath((string) $path, $maxBytes, $findings, $errors);
        }

        foreach ((array) $this->option('log') as $log) {
            $this->scanLog((string) $log, $findings, $errors);
        }

        $this->line((string) json_encode([
            'generated_at' => now()->toIso8601String(),
            'read_only'    => true,
            'findings'     => $findings,
            'errors'       => $errors,
        ], JSON_PRETTY_PRINT|JSON_UNESCAPED_SLASHES|JSON_THROW_ON_ERROR));

        return $findings === [] && $errors === [] ? self::SUCCESS : self::FAILURE;
    }

    /**
     * @param  array<int, array<string, mixed>>  $findings
     * @param  array<int, array<string, string>>  $errors
     */
    private function scanPath(string $path, int $maxBytes, array &$findings, array &$errors): void
    {
        $realPath = realpath($path);
        if ($realPath === false || ! is_dir($realPath)) {
            $errors[] = ['path' => $path, 'error' => 'Directory does not exist or is not readable.'];

            return;
        }

        $iterator = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($realPath, FilesystemIterator::SKIP_DOTS),
        );

        /** @var SplFileInfo $file */
        foreach ($iterator as $file) {
            if (! $file->isFile() || $file->isLink() || ! $file->isReadable()) {
                continue;
            }

            try {
                $hash = hash_file('sha256', $file->getPathname());
                $reasons = [];
                $extension = strtolower($file->getExtension());

                if (in_array($hash, self::SHIPPED_SECURITY_FILE_HASHES, true)) {
                    continue;
                }

                if (in_array($hash, self::MALWARE_HASHES, true)) {
                    $reasons[] = 'known-malware-hash';
                }

                if (in_array($extension, self::SUSPICIOUS_EXTENSIONS, true)
                    || in_array(strtolower($file->getFilename()), ['.htaccess', '.user.ini'], true)) {
                    $reasons[] = 'server-executable-or-configuration-extension';
                }

                if ($file->getSize() <= $maxBytes) {
                    $contents = file_get_contents($file->getPathname());
                    if (is_string($contents) && preg_match('/<\?(?:php|=)|\b(?:shell_exec|passthru|system|proc_open|popen)\s*\(/i', $contents)) {
                        $reasons[] = 'php-or-command-execution-content';
                    }
                }

                if ($reasons !== []) {
                    $findings[] = [
                        'type'     => 'file',
                        'path'     => $file->getPathname(),
                        'size'     => $file->getSize(),
                        'modified' => date(DATE_ATOM, $file->getMTime()),
                        'sha256'   => $hash,
                        'reasons'  => array_values(array_unique($reasons)),
                    ];
                }
            } catch (Throwable $exception) {
                $errors[] = ['path' => $file->getPathname(), 'error' => $exception->getMessage()];
            }
        }
    }

    /**
     * @param  array<int, array<string, mixed>>  $findings
     * @param  array<int, array<string, string>>  $errors
     */
    private function scanLog(string $log, array &$findings, array &$errors): void
    {
        $realPath = realpath($log);
        if ($realPath === false || ! is_file($realPath) || ! is_readable($realPath)) {
            $errors[] = ['path' => $log, 'error' => 'Log does not exist or is not readable.'];

            return;
        }

        $handle = fopen($realPath, 'rb');
        if ($handle === false) {
            $errors[] = ['path' => $log, 'error' => 'Log could not be opened.'];

            return;
        }

        $lineNumber = 0;
        while (($line = fgets($handle)) !== false) {
            $lineNumber++;
            $decoded = rawurldecode($line);
            if (preg_match('#(?:POST\s+/files/upload|/uploads/[^?\s]+\.php(?:\?|\s)|MAGICAI_RCE_VERIFIED|103\.130\.18\.(?:144|45)|116\.129\.177\.107|129\.226\.216\.22|2402:8780:1052:5d1c:65d2:9e91:b67f:42ac)#i', $decoded)) {
                $findings[] = [
                    'type'   => 'log',
                    'path'   => $realPath,
                    'line'   => $lineNumber,
                    'sample' => mb_substr(rtrim($line), 0, 500),
                ];
            }
        }

        fclose($handle);
    }
}
