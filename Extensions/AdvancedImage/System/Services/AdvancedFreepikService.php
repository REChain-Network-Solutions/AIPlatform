<?php

namespace App\Extensions\AdvancedImage\System\Services;

use App\Extensions\AdvancedImage\System\Services\Traits\UseImage;
use App\Models\UserOpenai;
use GuzzleHttp\Promise\PromiseInterface;
use Illuminate\Http\Client\Response;
use Illuminate\Support\Arr;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use RuntimeException;

class AdvancedFreepikService
{
    use UseImage;

    private const ENDPOINTS = [
        'reimagine'         => 'ai/beta/text-to-image/reimagine-flux',
        'style_transfer'    => 'https://api.freepik.com/v1/ai/image-style-transfer',
        //        'remove_background' => 'https://api.freepik.com/v1/ai/beta/remove-background',
        'upscale'           => 'ai/image-upscaler',
        'image-relight'		   => 'ai/image-relight',
    ];

    public function webhook(UserOpenai $task, array $data): void
    {
        if (data_get($data, 'status') === 'COMPLETED') {
            $image = Arr::first(data_get($data, 'generated'));

            $path = $this->downloadAndSaveImageFromUrl($image);

            $task->update([
                'title'   => str_replace('image-editor/', '', $path),
                'status'  => 'COMPLETED',
                'output'  => $path,
            ]);
        }
    }

    public function checkStatus(UserOpenai $task): UserOpenai
    {
        $payload = $task->payload;

        $response = $this->requestGet($payload['link']);

        if (! is_array($response) && $response->json('data.status') === 'COMPLETED') {
            $image = Arr::first($response->json('data.generated'));

            $path = $this->downloadAndSaveImageFromUrl($image);

            $task->update([
                'title'   => str_replace('image-editor/', '', $path),
                'status'  => 'COMPLETED',
                'output'  => $path,
            ]);
        }

        return $task;
    }

    public function generate(string $model, array $params): array
    {
        $response = match ($model) {
            'reimagine'         => $this->reimagine($params),
            'image_relight'     => $this->imageRelight($params),
            //            'remove_background' => $this->removeBackground($params),
            'upscale'           => $this->upscale($params),
            'style_transfer'    => $this->styleTransfer($params),
            default             => [
                'status'    => 'error',
                'message'   => 'Invalid model.',
            ],
        };

        return $response;
    }

    private function styleTransfer($params): PromiseInterface|array|Response
    {
        $image = $this->encodeFile($params['uploaded_image']);

        $reference_image = $this->encodeFile($params['reference_image']);

        $data = [
            'image'                  => $image,
            'reference_image'        => $reference_image,
            'webhook_url'            => config('app.url') . '/api/webhook/advanced-image/freepik',
            'prompt'                 => $params['description'] ?? 'enhance clarity sharp details',
            'style_strength'         => 85,
            'structure_strength'     => 60,
            'engine'                 => 'colorful_anime',
        ];

        $response = $this->requestPost(self::ENDPOINTS['style_transfer'], $data);

        if (is_array($response)) {
            return $response;
        }

        if ($response->json('task_id') && $response->json('task_status')) {
            return [
                'task_status' => $response->json('task_status'),
                'task_id'     => $response->json('task_id'),
                'status'      => 'success',
                'photo'       => null,
                'message'     => 'Task is processing.',
                'link'        => 'ai/image-style-transfer/' . $response->json('task_id'),
            ];
        }

        return [
            'task_status' => 'error',
            'task_id'     => null,
            'status'      => 'error',
            'message'     => 'Unable to process action.',
        ];
    }

    private function imageRelight(array $params)
    {
        request()->validate([
            'image_relight'                       => 'required',
            'description'                         => 'required',
        ]);

        $image = $this->encodeFile($params['uploaded_image']);

        $transfer_light_from_reference_image = $this->encodeFile($params['image_relight']);

        $data = [
            'image'                               => $image,
            'prompt'                              => $params['description'],
            'transfer_light_from_reference_image' => $transfer_light_from_reference_image,
            'light_transfer_strength'             => 100,
            'interpolate_from_original'           => false,
            'change_background'                   => true,
            'style'                               => $params['style'],
            'webhook_url'                         => config('app.url') . '/api/webhook/advanced-image/freepik',
        ];

        $response = $this->requestPost(self::ENDPOINTS['upscale'], $data);

        if (is_array($response)) {
            return $response;
        }

        if ($response->json('data.task_id') && $response->json('data.status')) {
            return [
                'task_status' => $response->json('data.status'),
                'task_id'     => $response->json('data.task_id'),
                'status'      => 'success',
                'photo'       => null,
                'message'     => 'Task is processing.',
                'link'        => 'ai/image-relight/' . $response->json('data.task_id'),
            ];
        }

        return [
            'task_status' => 'error',
            'task_id'     => null,
            'status'      => 'error',
            'message'     => 'Unable to process action.',
        ];
    }

    private function upscale($params): PromiseInterface|array|Response
    {
        $image = $this->encodeFile($params['uploaded_image']);

        $data = [
            'image'        => $image,
            'webhook_url'  => config('app.url') . '/api/webhook/advanced-image/freepik',
            'scale_factor' => '2x',
            'optimized_for'=> 'standard',
            'prompt'       => $params['description'] ?? 'enhance clarity sharp details',
            'creativity'   => 2,
            'hdr'          => 1,
            'resemblance'  => 0,
            'fractality'   => -1,
            'engine'       => 'magnific_sparkle',
        ];

        $response = $this->requestPost(self::ENDPOINTS['upscale'], $data);

        if (is_array($response)) {
            return $response;
        }

        if ($response->json('data.task_id') && $response->json('data.status')) {
            return [
                'task_status' => $response->json('data.status'),
                'task_id'     => $response->json('data.task_id'),
                'status'      => 'success',
                'photo'       => null,
                'message'     => 'Task is processing.',
                'link'        => 'ai/image-upscaler/' . $response->json('data.task_id'),
            ];
        }

        return [
            'task_status' => 'error',
            'task_id'     => null,
            'status'      => 'error',
            'message'     => 'Unable to process action.',
        ];
    }

    private function removeBackground($params) {}

    public function reimagine($params): string|array
    {
        request()->validate(['description' => 'required|string']);

        $image = $this->encodeFile($params['uploaded_image']);

        $data = [
            'image'        => $image,
            'prompt'       => $params['description'],
            'imagination'  => $params['imagination'] ?? 'wild',
            'aspect_ratio' => $params['aspect_ratio'] ?? 'square_1_1',
        ];

        $response = $this->requestPost(self::ENDPOINTS['reimagine'], $data);

        if (is_array($response)) {
            return $response;
        }

        if ($response->json('task_id') && $response->json('status') === 'COMPLETED') {
            $image = Arr::first($response->json('generated'));

            $path = $this->downloadAndSaveImageFromUrl($image);

            if ($path) {
                return [
                    'task_status' => $response->json('status'),
                    'status'      => 'success',
                    'task_id'     => $response->json('task_id'),
                    'photo'       => $path,
                    'link'        => 'ai/image-upscaler/' . $response->json('data.task_id'),
                ];
            }
        }

        return [
            'task_status' => 'error',
            'task_id'     => null,
            'status'      => 'error',
            'message'     => 'Unable to process action.',
        ];
    }

    public function downloadAndSaveImageFromUrl($url): ?string
    {
        // Resmin içeriğini çek
        $response = Http::get($url);

        if ($response->successful()) {
            // Dosya uzantısını tahmin et
            $extension = pathinfo(parse_url($url, PHP_URL_PATH), PATHINFO_EXTENSION) ?: 'jpg';

            // Benzersiz bir dosya adı oluştur
            $fileName = Str::uuid() . '.' . $extension;

            // Storage path (örnek: storage/app/public/images)
            $path = 'image-editor/' . $fileName;

            // Dosyayı kaydet
            Storage::disk('public')->put($path, $response->body());

            // İsteğe bağlı: URL'yi döndür (örnek: http://.../storage/images/abc.jpg)
            return $this->imagePath($path);
        }

        return null;
    }

    private function encodeFile($file): string
    {
        if (empty($file)) {
            throw new RuntimeException('Make selection on image first');
        }

        return base64_encode(file_get_contents($file->getRealPath()));
    }

    public function requestGet($url, $contentType = 'application/json'): PromiseInterface|array|Response
    {
        $http = Http::withHeaders([
            'Content-Type'      => $contentType,
            'Accept'            => 'application/json',
            'x-freepik-api-key' => $this->getApiKey(),
        ])
            ->baseUrl('https://api.freepik.com/v1/')
            ->get('ai/image-relight');

        if ($http->successful()) {
            return $http;
        }

        $error = $http->json('message');

        return [
            'status'    => 'error',
            'message'   => $error ?? 'Unable to process action.',
        ];
    }

    public function requestPost($url, $data, $contentType = 'application/json'): PromiseInterface|array|Response
    {
        $http = Http::withHeaders([
            'Content-Type'      => $contentType,
            'Accept'            => 'application/json',
            'x-freepik-api-key' => $this->getApiKey(),
        ])
            ->baseUrl('https://api.freepik.com/v1/')
            ->post($url, $data);

        if ($http->successful()) {
            return $http;
        }

        $error = $http->json('message');

        return [
            'status'    => 'error',
            'message'   => $error ?? 'Unable to process action.',
        ];
    }

    private function getApiKey(): ?string
    {
        return setting('freepik_api_key');
    }
}
