<?php

namespace App\Extensions\AdvancedImage\System\Services\Traits;

trait UseImage
{
    public function title(string $path): string
    {
        return str_replace(['/uploads/image-editor/', '/uploads', 'image-editor', '/'], '', $path);
    }

    public function imagePath(string $path, string $disk = 'uploads'): string
    {
        if ($disk === 'uploads') {
            return '/uploads/' . $path;
        }

        return $path;
    }
}
