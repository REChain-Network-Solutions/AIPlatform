<?php

namespace App\Extensions\AdvancedImage\System\Services;

use App\Extensions\AdvancedImage\System\Services\Traits\UseImage;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class ClipDropService
{
    use UseImage;

    public string $action;

    public ?UploadedFile $photo = null;

    public string $baseUrl = 'https://clipdrop-api.co/';

    public function generate(): array|string
    {
        $action = match ($this->getAction()) {
            'reimagine'          => $this->reimagine(),
            'remove_background'  => $this->remove_background(),
            'replace_background' => $this->replace_background(),
            'sketch_to_image'    => $this->sketch_to_image(),
            'text_to_image'      => $this->text_to_image(),
            'upscale'            => $this->upscale(),
            'remove_text'        => $this->remove_text(),
            'uncrop'             => $this->uncrop(),
            'cleanup'            => $this->cleanup(),
            'inpainting'         => $this->inpainting(),
            default              => [],
        };

        if (! is_array($action)) {
            $fileName = 'image-editor/' . Str::random('10') . '.jpg';
            Storage::disk('public')
                ->put(
                    $fileName,
                    $action->body()
                );

            return $this->imagePath($fileName);
        }

        return [
            'status'  => false,
            'message' => data_get($action, 'message', 'Something went wrong, please try again.'),
        ];
    }

    public function inpainting()
    {
        return $this->request(
            $this->baseUrl . 'text-inpainting/v1', [
                'text_prompt' => request('description'),
            ]
        );
    }

    public function cleanup()
    {
        return $this->request(
            $this->baseUrl . 'cleanup/v1',
        );
    }

    public function uncrop()
    {
        return $this->request(
            $this->baseUrl . 'uncrop/v1', [
                'extend_left'  => request('extend_left'),
                'extend_down'  => request('extend_down'),
                'extend_up'    => request('extend_up'),
                'extend_right' => request('extend_right'),
            ], 'image_file');
    }

    public function reimagine()
    {
        return $this->request(
            $this->baseUrl . 'reimagine/v1/reimagine'
        );
    }

    public function remove_text()
    {
        return $this->request(
            $this->baseUrl . 'remove-text/v1'
        );
    }

    public function upscale()
    {
        return $this->request(
            $this->baseUrl . 'image-upscaling/v1/upscale', [
                'target_width'  => request('target_width', '2048'),
                'target_height' => request('target_height', '2048'),
            ]);
    }

    public function remove_background()
    {
        return $this->request(
            $this->baseUrl . 'remove-background/v1', [], 'image_file', true);
    }

    public function replace_background()
    {
        return $this->request(
            $this->baseUrl . 'replace-background/v1', [
                'prompt' => request('description'),
            ]);
    }

    public function sketch_to_image()
    {
        return $this->request(
            $this->baseUrl . 'sketch-to-image/v1/sketch-to-image', [
                'prompt' => request('description'),
            ], 'sketch_file');
    }

    public function text_to_image()
    {
        return $this->request(
            $this->baseUrl . 'text-to-image/v1', [
                'prompt' => request('description'),
            ], 'sketch_file', false);
    }

    public function getAction(): string
    {
        return $this->action;
    }

    public function setAction(string $action): self
    {
        $this->action = $action;

        return $this;
    }

    public function getPhoto(): UploadedFile
    {
        return $this->photo;
    }

    public function setPhoto(?UploadedFile $photo = null): ClipDropService
    {
        $this->photo = $photo;

        return $this;
    }

    public function request(string $url, array $data = [], $fileKey = 'image_file', $hasFile = true)
    {
        $http = Http::asMultipart()
            ->withHeaders([
                'x-api-key' => $this->getApiKey(),
            ])
            ->when($hasFile, function ($http) use ($fileKey) {
                $http = $http->attach($fileKey, $this->getPhoto()->getContent(), $this->getPhoto()->getClientOriginalName());

                if (request()->hasFile('mask_file')) {
                    $maskFile = request()->file('mask_file');
                    $http = $http->attach('mask_file', $maskFile->getContent(), $maskFile->getClientOriginalName());
                }

                return $http;
            })
            ->post($url, $data);

        if ($http->successful()) {
            return $http;
        }

        $error = $http->json('error');

        if ($error) {
            if ($error === 'The request is not valid: Input image must have a square ratio, ie have the same height and width.') {
                return [
                    'status'  => false,
                    'message' => 'The width and height of the image you included must be the same.',
                ];
            }

            return [
                'status'  => false,
                'message' => $error,
            ];
        }

        return [
            'status'  => false,
            'message' => 'Something went wrong, please try again.',
        ];
    }

    public function getApiKey(): string
    {
        return setting('clipdrop_api_key', '');
    }
}
