<?php

namespace App\Extensions\AdvancedImage\System\Services;

use App\Extensions\AdvancedImage\System\Services\Traits\UseImage;
use App\Models\UserOpenai;
use Exception;
use GuzzleHttp\Client;
use Illuminate\Support\Arr;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use RuntimeException;

class AdvancedNovitaService
{
    use UseImage;

    private Client $client;

    // API Endpoints
    private const ENDPOINTS = [
        'remove_background'  => 'https://api.novita.ai/v3/remove-background',
        'merge_face'         => 'https://api.novita.ai/v3/merge-face',
        'replace_background' => 'https://api.novita.ai/v3/replace-background',
        'upscale'            => 'https://api.novita.ai/v3/async/upscale',
        'reimagine'          => 'https://api.novita.ai/v3/reimagine',
        'remove_text'        => 'https://api.novita.ai/v3/remove-text',
        'cleanup'            => 'https://api.novita.ai/v3/cleanup',
        'text_to_image'      => 'https://api.novita.ai/v3/async/txt2img',
        'inpainting'         => 'https://api.novita.ai/v3/async/inpainting',
        'check_status'       => 'https://api.novita.ai/v3/async/task-result?task_id=',
    ];

    public function __construct()
    {
        $this->client = new Client;
    }

    public function webhook(UserOpenai $task, array $data): void
    {
        $image = data_get($data, '0.image_url');

        if ($image) {
            $path = $this->downloadAndSaveImageFromUrl($image);

            $task->update([
                'title'   => $this->title($path),
                'status'  => 'COMPLETED',
                'output'  => $path,
            ]);
        }
    }

    public function generate(string $model, array $params): array
    {
        $action = $this->handleAction($model, $params);

        if (isset($action['task']['task_id'])) {

            if ($model === 'merge_face') {
                $action['image_type'] = 'png';
            }

            return $this->processTaskWithImage($action);
        }

        if (isset($action['task_id'])) {
            return [
                'task_id'     => $action['task_id'],
                'photo'       => '',
                'status'      => 'success',
                'task_status' => 'IN_PROGRESS',
                'message'     => 'Task is processing.',
                'link'        => 'https://api.novita.ai/v3/async/task-result?task_id=' . $action['task_id'],
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
            'inpainting'         => $this->inpainting($params),
            'merge_face'         => $this->merge_face($params),
            default              => []
        };
    }

    private function processTaskWithImage(array $action): array
    {
        if (isset($action['image_file'])) {

            $imageType = $action['image_type'] ?? 'jpg';

            $contents = base64_decode($action['image_file']);

            $fileName = 'image-editor/' . Str::random(10) . '.' . $imageType;

            Storage::disk('public')->put($fileName, $contents);

            return [
                'task_id' => $action['task']['task_id'],
                'photo'   => $this->imagePath($fileName),
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
        try {
            $response = $this->client->post($endpoint, [
                'headers' => [
                    'Authorization' => 'Bearer ' . setting('novita_api_key'),
                    'Content-Type'  => 'application/json',
                ],
                'json' => $payload,
            ]);

            return json_decode($response->getBody()->getContents(), true);
        } catch (Exception $exception) {
            if (! $exception->hasResponse()) {
                return ['type' => 'error', 'message' => __('Unknown error: No response from the server.')];
            }

            $responseBody = $exception->getResponse()->getBody();
            $errorResponse = json_decode($responseBody, true);
            if (! is_array($errorResponse)) {
                return ['type' => 'error', 'message' => __('Unknown error: Invalid response format.')];
            }
            $message = $errorResponse['message'] ?? __('Unknown error');
            $metadataMessage = '';
            if (isset($errorResponse['metadata']) && is_array($errorResponse['metadata'])) {
                $metadataMessage = $errorResponse['metadata']['image_file'] ?? '';
            }

            return ['type' => 'error', 'message' => trim($message . '. ' . $metadataMessage)];
        }
    }

    private function inpainting(array $params): array
    {
        request()->validate([
            'description' => 'required|string|max:1024',
        ], [
            'description.required' => 'Please provide a prompt',
        ]);

        $image = $this->encodeFile($params['photo']);
        $mask = $this->encodeFile($params['mask']);

        return $this->apiRequest(self::ENDPOINTS['inpainting'], [
            'extra'   => [
                'response_image_type' => 'jpeg',
                'webhook'             => [
                    'url'  => config('app.url') . '/api/webhook/advanced-image/novita',
                ],
                'enable_nsfw_detection' => false,
            ],
            'request' => [
                'model_name'                 => 'realisticVisionV40_v40VAE-inpainting_81543.safetensors',
                'prompt'                     => $params['description'],
                'negative_prompt'            => '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, BadDream, UnrealisticDream',
                'image_num'                  => 1,
                'steps'                      => 25,
                'seed'                       => -1,
                'clip_skip'                  => 1,
                'guidance_scale'             => 7.5,
                'sampler_name'               => 'Euler a',
                'mask_blur'                  => 1,
                'inpainting_full_res'        => 1,
                'inpainting_full_res_padding'=> 32,
                'inpainting_mask_invert'     => 0,
                'initial_noise_multiplier'   => 1,
                'strength'                   => 0.85,
                'image_base64'               => $image,
                'mask_image_base64'          => $mask,
            ],
        ]);
    }

    private function merge_face(array $params): array
    {
        $image = $this->encodeFile($params['photo']);
        $faceImage = $this->encodeFile(request()->file('face_image'));

        return $this->apiRequest(self::ENDPOINTS['merge_face'], [
            'image_file'      => $image,
            'face_image_file' => $faceImage,
            'extra'           => [
                'response_image_type' => 'png',
            ],
        ]);
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
        if (empty($file)) {
            throw new RuntimeException('Make selection on image first');
        }

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

    public function checkStatus(UserOpenai $task): UserOpenai
    {

        $http = Http::withHeaders([
            'Authorization' => 'Bearer ' . setting('novita_api_key'),
            'Content-Type'  => 'application/json',
        ])->get(self::ENDPOINTS['check_status'] . $task['request_id']);

        if ($http->json('task.status') === 'TASK_STATUS_SUCCEED' && is_array($http->json('images'))) {
            $image = Arr::first($http->json('images'));

            $path = $this->downloadAndSaveImageFromUrl($image['image_url']);

            $task->update([
                'title'   => $this->title($path),
                'status'  => 'COMPLETED',
                'output'  => $path,
            ]);

        }

        return $task;
    }

    public function downloadAndSaveImageFromUrl($url): ?string
    {
        $response = Http::get($url);

        if ($response->successful()) {

            $extension = pathinfo(parse_url($url, PHP_URL_PATH), PATHINFO_EXTENSION) ?: 'jpg';

            $fileName = Str::uuid() . '.' . $extension;

            $path = 'image-editor/' . $fileName;

            Storage::disk('uploads')->put($path, $response->body());

            return $this->imagePath($path);
        }

        return null;
    }
}
