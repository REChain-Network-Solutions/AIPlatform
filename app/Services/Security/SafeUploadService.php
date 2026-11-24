<?php

declare(strict_types=1);

namespace App\Services\Security;

use App\Models\SettingTwo;
use App\Models\User;
use App\Models\UserUpload;
use finfo;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Illuminate\Validation\ValidationException;
use RuntimeException;
use ZipArchive;

class SafeUploadService
{
    private const MAX_BYTES = 20 * 1024 * 1024;

    private const MIME_EXTENSIONS = [
        'application/pdf'                                                         => ['pdf'],
        'application/msword'                                                      => ['doc'],
        'application/vnd.ms-excel'                                                => ['xls'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document' => ['docx'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'       => ['xlsx'],
        'image/jpeg'                                                              => ['jpg', 'jpeg'],
        'image/png'                                                               => ['png'],
        'image/gif'                                                               => ['gif'],
        'image/webp'                                                              => ['webp'],
        'audio/mpeg'                                                              => ['mp3'],
        'audio/wav'                                                               => ['wav'],
        'audio/x-wav'                                                             => ['wav'],
        'audio/mp4'                                                               => ['m4a'],
        'video/mp4'                                                               => ['mp4'],
        'video/quicktime'                                                         => ['mov'],
        'video/x-msvideo'                                                         => ['avi'],
        'video/x-ms-wmv'                                                          => ['wmv'],
        'video/webm'                                                              => ['webm'],
        'text/plain'                                                              => ['txt'],
        'text/csv'                                                                => ['csv'],
    ];

    private const DEFAULT_EXTENSIONS = [
        'jpg', 'jpeg', 'png', 'gif', 'webp', 'mp4', 'avi', 'mov', 'wmv', 'webm',
        'mp3', 'wav', 'm4a', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'txt', 'csv',
    ];

    /**
     * @param  array{name: string, data: string}  $payload
     */
    public function store(User $user, array $payload): UserUpload
    {
        $name = trim($payload['name']);
        if (str_contains($name, '/') || str_contains($name, '\\')) {
            throw ValidationException::withMessages(['files' => __('File paths are not allowed in upload names.')]);
        }

        $extension = strtolower(pathinfo($name, PATHINFO_EXTENSION));

        if (! $this->hasSafeName($name, $extension)) {
            throw ValidationException::withMessages(['files' => __('The file name or extension is not allowed.')]);
        }

        if (! preg_match('/^data:([a-z0-9.+-]+\/[a-z0-9.+-]+);base64,([a-zA-Z0-9+\/=\r\n]+)$/i', $payload['data'], $matches)) {
            throw ValidationException::withMessages(['files' => __('The file payload must be valid base64 data.')]);
        }

        $contents = base64_decode(preg_replace('/\s+/', '', $matches[2]), true);

        if ($contents === false || $contents === '' || strlen($contents) > self::MAX_BYTES) {
            throw ValidationException::withMessages(['files' => __('The decoded file is invalid or exceeds 20 MB.')]);
        }

        if (preg_match('/<\?(?:php|=)/i', $contents)) {
            throw ValidationException::withMessages(['files' => __('Executable content is not allowed in uploads.')]);
        }

        $mimeType = (new finfo(FILEINFO_MIME_TYPE))->buffer($contents) ?: 'application/octet-stream';
        $mimeType = $this->normalizeOfficeMime($contents, $mimeType, $extension);
        $allowedExtensions = $this->allowedExtensions();

        if (! in_array($extension, $allowedExtensions, true)) {
            throw ValidationException::withMessages(['files' => __('The file content does not match an allowed type.')]);
        }

        if (in_array($extension, ['txt', 'csv'], true)) {
            $this->assertPlainText($contents, $extension);
            $mimeType = $extension === 'csv' ? 'text/csv' : 'text/plain';
        } else {
            if (! isset(self::MIME_EXTENSIONS[$mimeType])
                || ! in_array($extension, self::MIME_EXTENSIONS[$mimeType], true)) {
                throw ValidationException::withMessages(['files' => __('The file content does not match an allowed type.')]);
            }

            if ($extension === 'pdf') {
                $this->assertPdfStructure($contents);
            }
        }

        if (str_starts_with($mimeType, 'image/')) {
            $contents = $this->reencodeImage($contents, $extension);
        }

        $id = (string) Str::ulid();
        $configuredDisk = SettingTwo::getCache()->ai_image_storage ?? 'local';
        $disk = in_array($configuredDisk, ['s3', 'r2'], true)
            ? $configuredDisk
            : 'local';
        $path = sprintf('user-uploads/%d/%s.%s', $user->id, $id, $extension === 'jpeg' ? 'jpg' : $extension);

        if (! Storage::disk($disk)->put($path, $contents, ['visibility' => 'private'])) {
            throw new RuntimeException('The upload could not be stored.');
        }

        return UserUpload::query()->create([
            'id'            => $id,
            'user_id'       => $user->id,
            'disk'          => $disk,
            'path'          => $path,
            'original_name' => $name,
            'mime_type'     => $mimeType,
            'extension'     => $extension === 'jpeg' ? 'jpg' : $extension,
            'size'          => strlen($contents),
            'sha256'        => hash('sha256', $contents),
        ]);
    }

    private function assertPlainText(string $contents, string $extension): void
    {
        $probe = $contents;

        if (str_starts_with($probe, "\xFF\xFE") || str_starts_with($probe, "\xFE\xFF")) {
            $converted = mb_convert_encoding($probe, 'UTF-8', 'UTF-16');

            if ($converted !== false && $converted !== '') {
                $probe = $converted;
            }
        }

        if (str_contains($probe, "\0")) {
            throw ValidationException::withMessages(['files' => __('The file content does not match an allowed type.')]);
        }

        $encoding = mb_detect_encoding($probe, ['UTF-8', 'ASCII', 'ISO-8859-1'], true);

        if ($encoding === false) {
            throw ValidationException::withMessages(['files' => __('The file content does not match an allowed type.')]);
        }
    }

    private function assertPdfStructure(string $contents): void
    {
        $contents = ltrim($contents, "\xEF\xBB\xBF \t\r\n");

        if (! str_starts_with($contents, '%PDF-')) {
            throw ValidationException::withMessages(['files' => __('The file content does not match an allowed type.')]);
        }

        $tail = substr($contents, -1024);

        if ($tail === false || ! str_contains($tail, '%%EOF')) {
            throw ValidationException::withMessages(['files' => __('The file content does not match an allowed type.')]);
        }
    }

    private function hasSafeName(string $name, string $extension): bool
    {
        if ($name === '' || str_starts_with($name, '.') || substr_count($name, '.') !== 1) {
            return false;
        }

        return preg_match('/^[a-z0-9]{1,10}$/', $extension) === 1
            && ! preg_match('/(?:php\d*|phtml|pht|phar|phps|cgi|shtml|asp|aspx|jsp|htaccess|user\.ini)/i', $name);
    }

    /** @return array<int, string> */
    private function allowedExtensions(): array
    {
        $configured = array_filter(array_map(
            static fn (string $extension): string => strtolower(trim($extension)),
            explode(',', (string) setting('media_allowed_types', implode(',', self::DEFAULT_EXTENSIONS))),
        ));

        return array_values(array_intersect(self::DEFAULT_EXTENSIONS, $configured));
    }

    private function normalizeOfficeMime(string $contents, string $mimeType, string $extension): string
    {
        if (! in_array($mimeType, ['application/zip', 'application/x-zip-compressed'], true)
            || ! in_array($extension, ['docx', 'xlsx'], true)) {
            return $mimeType;
        }

        $temporary = tmpfile();
        if ($temporary === false) {
            return $mimeType;
        }

        fwrite($temporary, $contents);
        $metadata = stream_get_meta_data($temporary);
        $zip = new ZipArchive;

        if ($zip->open($metadata['uri']) !== true || $zip->locateName('[Content_Types].xml') === false) {
            fclose($temporary);

            return $mimeType;
        }

        $expectedPrefix = $extension === 'docx' ? 'word/' : 'xl/';
        $hasExpectedDirectory = false;
        for ($index = 0; $index < $zip->numFiles; $index++) {
            if (str_starts_with((string) $zip->getNameIndex($index), $expectedPrefix)) {
                $hasExpectedDirectory = true;

                break;
            }
        }
        $zip->close();
        fclose($temporary);

        if (! $hasExpectedDirectory) {
            return $mimeType;
        }

        return $extension === 'docx'
            ? 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    }

    private function reencodeImage(string $contents, string $extension): string
    {
        $image = @imagecreatefromstring($contents);
        if ($image === false) {
            throw ValidationException::withMessages(['files' => __('The image could not be decoded safely.')]);
        }

        ob_start();
        $written = match ($extension) {
            'jpg', 'jpeg' => imagejpeg($image, null, 90),
            'png'         => imagepng($image),
            'gif'         => imagegif($image),
            'webp'        => imagewebp($image, null, 90),
            default       => false,
        };
        $reencoded = ob_get_clean();
        imagedestroy($image);

        if (! $written || ! is_string($reencoded)) {
            throw ValidationException::withMessages(['files' => __('The image could not be encoded safely.')]);
        }

        return $reencoded;
    }
}
