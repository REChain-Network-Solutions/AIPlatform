<?php

namespace App\Console\Commands;

use App\Models\User;
use App\Models\UserOpenaiChat;
use App\Models\UserOpenaiChatMessage;
use App\Services\Security\SafeUploadService;
use finfo;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;
use Illuminate\Validation\ValidationException;
use RuntimeException;
use Symfony\Component\Finder\Finder;
use Throwable;

class MigrateLegacyUploads extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'security:migrate-legacy-uploads
        {--path= : Legacy media directory; defaults to public/uploads/media}
        {--apply : Move files and update database references instead of producing a dry-run report}';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Inventory legacy public uploads, quarantine suspicious files, and migrate safe files to private storage';

    /**
     * Execute the console command.
     */
    public function handle(SafeUploadService $safeUploadService): int
    {
        $root = realpath((string) ($this->option('path') ?: public_path('uploads/media')));
        if ($root === false || ! is_dir($root)) {
            $this->error('The legacy upload directory does not exist.');

            return self::FAILURE;
        }

        $apply = (bool) $this->option('apply');
        $results = [];

        foreach ((new Finder)->files()->ignoreDotFiles(false)->in($root) as $file) {
            $path = $file->getRealPath();
            if ($path === false || is_link($path)) {
                continue;
            }

            $relativePath = ltrim(str_replace($root, '', $path), DIRECTORY_SEPARATOR);
            $contents = file_get_contents($path);
            $userId = preg_match('#(?:^|/)u-(\d+)(?:/|$)#', str_replace('\\', '/', $relativePath), $matches)
                ? (int) $matches[1]
                : 0;
            $suspicious = ! is_string($contents)
                || preg_match('/<\?(?:php|=)|\b(?:shell_exec|passthru|system|proc_open|popen)\s*\(/i', $contents)
                || preg_match('/\.(?:php\d*|phtml|pht|phar|phps|cgi|shtml|asp|aspx|jsp|htaccess|user\.ini)$/i', $file->getFilename());

            $result = [
                'path'   => $path,
                'sha256' => hash_file('sha256', $path),
                'action' => $suspicious ? 'quarantine' : 'migrate',
                'status' => $apply ? 'pending' : 'dry-run',
            ];

            if (! $apply) {
                $results[] = $result;

                continue;
            }

            if (! is_string($contents)) {
                $result['status'] = 'error';
                $result['error'] = 'The source file could not be read; it was not modified.';
                $results[] = $result;

                continue;
            }

            try {
                if ($suspicious || $userId === 0 || ! ($user = User::query()->find($userId))) {
                    $quarantinePath = 'security-quarantine/' . $result['sha256'] . '-' . basename($path);
                    if (! Storage::disk('local')->put($quarantinePath, $contents)) {
                        throw new RuntimeException('Could not write the quarantine copy.');
                    }
                    if (! unlink($path)) {
                        Storage::disk('local')->delete($quarantinePath);

                        throw new RuntimeException('Could not remove the source after creating the quarantine copy.');
                    }
                    $result['status'] = 'quarantined';
                    $result['destination'] = $quarantinePath;
                } else {
                    $mime = (new finfo(FILEINFO_MIME_TYPE))->buffer($contents) ?: 'application/octet-stream';
                    $upload = $safeUploadService->store($user, [
                        'name' => $file->getFilename(),
                        'data' => sprintf('data:%s;base64,%s', $mime, base64_encode($contents)),
                    ]);
                    $oldPublicPath = '/uploads/media/' . str_replace('\\', '/', $relativePath);
                    $newPath = route('uploads.show', $upload);

                    UserOpenaiChatMessage::query()->where('pdfPath', $oldPublicPath)->update(['pdfPath' => $newPath]);
                    UserOpenaiChat::query()->where('reference_url', $oldPublicPath)->update(['reference_url' => $newPath]);
                    if (! unlink($path)) {
                        $upload->delete();
                        Storage::disk($upload->disk)->delete($upload->path);

                        throw new RuntimeException('Could not remove the source after migrating it.');
                    }

                    $result['status'] = 'migrated';
                    $result['upload_id'] = $upload->id;
                }
            } catch (ValidationException $exception) {
                try {
                    $quarantinePath = 'security-quarantine/' . $result['sha256'] . '-' . basename($path);
                    if (! Storage::disk('local')->put($quarantinePath, $contents)) {
                        throw new RuntimeException('Could not write the quarantine copy.');
                    }
                    if (! unlink($path)) {
                        Storage::disk('local')->delete($quarantinePath);

                        throw new RuntimeException('Could not remove the source after creating the quarantine copy.');
                    }
                    $result['status'] = 'quarantined';
                    $result['destination'] = $quarantinePath;
                    $result['reason'] = 'The file did not pass the current safe-upload policy.';
                } catch (Throwable $quarantineException) {
                    $result['status'] = 'error';
                    $result['error'] = $quarantineException->getMessage();
                }
            } catch (Throwable $exception) {
                $result['status'] = 'error';
                $result['error'] = $exception->getMessage();
            }

            $results[] = $result;
        }

        $this->line((string) json_encode([
            'apply'   => $apply,
            'results' => $results,
        ], JSON_PRETTY_PRINT|JSON_UNESCAPED_SLASHES|JSON_THROW_ON_ERROR));

        return collect($results)->contains('status', 'error') ? self::FAILURE : self::SUCCESS;
    }
}
