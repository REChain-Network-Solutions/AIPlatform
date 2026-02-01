<?php

namespace App\Extensions\AIRealtimeImage\System\Services;

use App\Domains\Entity\Enums\EntityEnum;
use App\Extensions\AIRealtimeImage\System\Enums\Status;
use App\Extensions\AIRealtimeImage\System\Models\RealtimeImage;
use App\Models\SettingTwo;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;

class TogetherService
{
    public string $link = 'https://api.together.xyz/v1/images/generations';

    public function generate(RealtimeImage $realtimeImage)
    {
        $prompt = $realtimeImage->getAttribute('prompt');
        $style = $realtimeImage->getAttribute('style');

        if ($style && $style !== 'none' && $style !== '') {
            $prompt .= '. Use ' . $style . ' style for the image.';
        }

        $http = Http::withHeaders([
            'Authorization'     => 'Bearer ' . $this->getApiKey(),
            'x-ratelimit-limit' => 10,
        ])
            ->post($this->link, [
                'prompt' => $prompt,
                'model'  => EntityEnum::fromSlug($realtimeImage->getAttribute('model'))?->value,
                'width'  => 1024,
                'height' => 768,
                'steps'  => 3,
            ]);

        $realtimeImage->update(['response' => $http->json()]);

        if ($http->successful()) {
            $this->success($realtimeImage);
        } else {
            $realtimeImage->update([
                'status'   => Status::failed,
            ]);
        }

        return $realtimeImage;
    }

    public function success(RealtimeImage $realtimeImage): true|RealtimeImage
    {
        $response = $realtimeImage->getAttribute('response');

        if ($response && isset($response['data'][0]['url'])) {

            $url = $response['data'][0]['url'];

            $url = $this->downloadImageToStorage($url);

            $realtimeImage->update([
                'image'  => $url,
                'status' => Status::success,
            ]);

            return $realtimeImage;
        }

        $realtimeImage->update([
            'status' => Status::failed,
        ]);

        return $realtimeImage;
    }

    public function downloadImageToStorage($url = null, $filename = null)
    {
        if (! $url) {
            return null;
        }
        $response = Http::get($url);
        if ($response->successful()) {
            $fileContent = $response->body();
            $extension = 'jpeg';
            if (! $filename) {
                $filename = uniqid('image_', true) . '.' . $extension;
            } else {
                $filename .= '.' . $extension;
            }
            $image_storage = SettingTwo::getCache()?->getAttribute('ai_image_storage');
            if ($image_storage === 'r2') {
                Storage::disk('r2')->put($filename, $fileContent);

                return Storage::disk('r2')->url($filename);
            }
            if ($image_storage === 's3') {
                Storage::disk('s3')->put($filename, $fileContent);

                return Storage::disk('s3')->url($filename);
            }
            $dump = Storage::disk('public')->put($filename, $fileContent);
            if ($dump) {
                return '/uploads/' . $filename;
            }

            return 'error';
        }

        return null;
    }

    public function getApiKey(): string
    {
        return setting('together_api_key', '');
    }
}
