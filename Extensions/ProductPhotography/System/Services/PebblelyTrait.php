<?php

namespace App\Extensions\ProductPhotography\System\Services;

use App\Helpers\Classes\Helper;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

trait PebblelyTrait
{
    public function getThemes(): array
    {
        return $this->get(self::THEMES_URL);
    }

    public function removeBg($image): array|string
    {
        $image = $this->imageControl($image);

        $body = [
            'image' => $image,
        ];

        $response = $this->post(self::REMOVE_BG_URL, $body);

        return $this->getResponse($response);
    }

    public function createBg($image, string $theme, int $height = 2048, int $width = 2048): array|string
    {
        $image = $this->imageControl($image);

        $body = [
            'images' => [$image],
            'theme'  => $theme,
            'height' => $height,
            'width'  => $width,
        ];

        $response = $this->post(self::CREATE_BACKGROUND_URL, $body);

        if (isset($response['error'])) {
            return [
                'error'   => $response['error'],
                'message' => $response['message'],
            ];
        } else {
            $imageBytes = base64_decode($response['data']);
            $imageName = Str::random(12) . '.png';

            Storage::disk($this->storage)->put($imageName, $imageBytes);

            if ($this->storage == 'public') {
                return Helper::parseUrl(config('app.url'), 'uploads', $imageName);
            } else {
                return Storage::disk($this->storage)->url($imageName);
            }
        }
    }

    private function imageControl($image): string
    {
        $imageContent = '';

        if ($image instanceof \Illuminate\Http\UploadedFile) {
            $imageContent = file_get_contents($image->getRealPath());
        } elseif (is_array($image) && isset($image['tmp_name'])) {
            $imageContent = file_get_contents($image['tmp_name']);
        } elseif (is_string($image) && is_readable($image)) {
            $imageContent = file_get_contents($image);
        }

        return base64_encode($imageContent);
    }

    private function imageSave($response): string
    {
        $imageBytes = base64_decode($response['data']);
        $imageName = Str::random(12) . '.png';

        Storage::disk('public')->put($imageName, $imageBytes);

        return Helper::parseUrl('/uploads', $imageName);
    }

    private function getResponse($response): array|string
    {
        if (isset($response['error'])) {
            return [
                'error'   => $response['error'],
                'message' => $response['message'],
            ];
        } else {
            return $this->imageSave($response);
        }
    }
}
