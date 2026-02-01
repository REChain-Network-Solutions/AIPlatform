<?php

namespace App\Extensions\PhotoStudio\System\Services;

use Exception;
use GuzzleHttp\Client;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class NovitaService
{
    private Client $client;

    // API Endpoints
    private const ENDPOINTS = [
        'remove_background'  => 'https://api.novita.ai/v3/remove-background',
        'replace_background' => 'https://api.novita.ai/v3/replace-background',
        'upscale'            => 'https://api.novita.ai/v3/async/upscale',
        'reimagine'          => 'https://api.novita.ai/v3/reimagine',
        'remove_text'        => 'https://api.novita.ai/v3/remove-text',
        'cleanup'            => 'https://api.novita.ai/v3/cleanup',
        'text_to_image'      => 'https://api.novita.ai/v3/async/txt2img',
        'check_status'       => 'https://api.novita.ai/v3/async/task-result?task_id=',
    ];

    public function __construct()
    {
        $this->client = new Client;
    }

    public function generate(string $model, array $params): array
    {
        $action = $this->handleAction($model, $params);

        if (isset($action['task']['task_id'])) {
            return $this->processTaskWithImage($action);
        }

        if (isset($action['task_id'])) {
            return [
                'task_id' => $action['task_id'],
                'photo'   => null,
                'status'  => 'in_progress',
            ];
        }

        return [
            'status'  => 'error',
            'message' => $action['message'] ?? 'Unable to process action.',
        ];
    }

    private function handleAction(string $model, array $params): array
    {
        return match ($model) {
            'reimagine'          => $this->reimagine($params),
            'remove_background'  => $this->removeBackground($params),
            'replace_background' => $this->replaceBackground($params),
            'text_to_image'      => $this->textToImage($params),
            'upscale'            => $this->upscale($params),
            'remove_text'        => $this->removeText($params),
            'cleanup'            => $this->cleanup($params),
            default              => []
        };
    }

    private function processTaskWithImage(array $action): array
    {
        if (isset($action['image_file'])) {
            $contents = base64_decode($action['image_file']);
            $fileName = 'photo-studio/' . Str::random(10) . '.jpg';

            Storage::disk('public')->put($fileName, $contents);

            return [
                'task_id' => $action['task']['task_id'],
                'photo'   => $fileName,
                'status'  => 'completed',
            ];
        }

        return [
            'task_id' => $action['task']['task_id'],
            'photo'   => null,
            'status'  => 'completed',
        ];
    }

    private function apiRequest(string $endpoint, array $payload): array
    {
        $response = $this->client->post($endpoint, [
            'headers' => [
                'Authorization' => 'Bearer ' . setting('novita_api_key'),
                'Content-Type'  => 'application/json',
            ],
            'json' => $payload,
        ]);

        return json_decode($response->getBody()->getContents(), true);
    }

    private function reimagine(array $params): array
    {
        $image = $this->encodeFile($params['photo']);

        return $this->apiRequest(self::ENDPOINTS['reimagine'], [
            'image_file' => $image,
            'extra'      => ['response_image_type' => 'png'],
        ]);
    }

    private function removeBackground(array $params): array
    {
        $image = $this->encodeFile($params['photo']);

        return $this->apiRequest(self::ENDPOINTS['remove_background'], [
            'image_file' => $image,
            'extra'      => ['response_image_type' => 'png'],
        ]);
    }

    private function replaceBackground(array $params): array
    {
        $image = $this->encodeFile($params['photo']);

        return $this->apiRequest(self::ENDPOINTS['replace_background'], [
            'image_file' => $image,
            'prompt'     => $params['description'],
            'extra'      => ['response_image_type' => 'png'],
        ]);
    }

    private function textToImage(array $params): array
    {
        return $this->apiRequest(self::ENDPOINTS['text_to_image'], [
            'request' => [
                'model_name'     => 'sd_xl_base_1.0.safetensors',
                'prompt'         => $params['description'],
                'width'          => 1024,
                'height'         => 1024,
                'image_num'      => 1,
                'steps'          => 20,
                'seed'           => 123,
                'clip_skip'      => 1,
                'guidance_scale' => 7.5,
            ],
            'extra' => ['response_image_type' => 'jpeg', 'enable_nsfw_detection' => false],
        ]);
    }

    private function upscale(array $params): array
    {
        $image = $this->encodeFile($params['photo']);

        return $this->apiRequest(self::ENDPOINTS['upscale'], [
            'request' => [
                'model_name'   => 'RealESRGAN_x4plus_anime_6B',
                'image_base64' => $image,
                'scale_factor' => 2,
            ],
            'extra' => ['response_image_type' => 'jpeg'],
        ]);
    }

    private function removeText(array $params): array
    {
        $image = $this->encodeFile($params['photo']);

        return $this->safeApiRequest(self::ENDPOINTS['remove_text'], [
            'image_file' => $image,
            'extra'      => ['response_image_type' => 'png'],
        ]);
    }

    private function cleanup(array $params): array
    {
        $image = $this->encodeFile($params['photo']);
        $mask = $this->encodeFile($params['mask']);

        return $this->apiRequest(self::ENDPOINTS['cleanup'], [
            'image_file' => $image,
            'mask_file'  => $mask,
            'extra'      => ['response_image_type' => 'png'],
        ]);
    }

    private function encodeFile($file): string
    {
        return base64_encode(file_get_contents($file->getRealPath()));
    }

    private function safeApiRequest(string $endpoint, array $payload): array
    {
        try {
            return $this->apiRequest($endpoint, $payload);
        } catch (Exception $exception) {
            if ($exception->hasResponse()) {
                $responseBody = $exception->getResponse()->getBody()->getContents();
                $decodedResponse = json_decode($responseBody, true);
                $errorMessage = $decodedResponse['message'] ?? 'API error';

                return ['type' => 'error', 'message' => $errorMessage];
            } else {
                return ['type' => 'error', 'message' => $exception->getMessage()];

            }
        }
    }

    public function checkStatus(string $taskId): array
    {
        $response = $this->client->get(self::ENDPOINTS['check_status'] . $taskId, [
            'headers' => [
                'Authorization' => 'Bearer ' . setting('novita_api_key'),
                'Content-Type'  => 'application/json',
            ],
        ]);

        return json_decode($response->getBody()->getContents(), true);
    }
}
