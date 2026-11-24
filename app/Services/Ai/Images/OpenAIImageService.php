<?php

namespace App\Services\Ai\Images;

use App\Domains\Entity\Enums\EntityEnum;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\DefaultOpenAiImageModel;
use App\Services\Ai\Images\Contracts\ImageGeneratorInterface;
use App\Services\Ai\OpenAI\Image\CreateImageEditService;
use App\Services\Ai\OpenAI\Image\CreateImageService;
use BadMethodCallException;
use InvalidArgumentException;

class OpenAIImageService implements ImageGeneratorInterface
{
    public function generate(array $options): array
    {
        $model = EntityEnum::tryFrom(str_replace('__', '.', (string) ($options['model'] ?? '')))
            ?? DefaultOpenAiImageModel::FALLBACK;
        $prompt = $options['prompt'] ?? throw new InvalidArgumentException(__('Prompt is required'));
        $imageReference = $options['image_reference'] ?? null;
        $background = $options['background'] ?? null;
        $size = $options['aspect_ratio'] ?? null;
        $quality = $options['quality'] ?? null;

        // Build the full prompt with attributes
        $fullPrompt = $this->buildPrompt($prompt, $options);

        $data = [
            'model'  => $model->value,
            'prompt' => $fullPrompt,
        ];

        if ($size) {
            $data['size'] = $size;
        }

        if ($background) {
            $data['background'] = $background;
        }

        if ($quality) {
            $data['quality'] = $quality;
        }

        return $this->generateImages($model, $prompt, $data, $imageReference);
    }

    public function supportsAsync(): bool
    {
        return false;
    }

    public function checkStatus(string $requestId, EntityEnum $entityEnum): array
    {
        throw new BadMethodCallException('OpenAI does not support async generation');
    }

    protected function buildPrompt(string $basePrompt, array $options): string
    {
        $attributes = [];

        if (! empty($options['image_style'])) {
            $attributes[] = "{$options['image_style']} style";
        }

        if (! empty($options['lighting'])) {
            $attributes[] = "{$options['lighting']} lighting";
        }

        if (! empty($options['mood'])) {
            $attributes[] = "{$options['mood']} mood";
        }

        return trim($basePrompt . ' ' . implode(' ', $attributes));
    }

    protected function generateImages(EntityEnum $model, string $prompt, array $data, string|array|null $imageRef = null): array
    {
        $imageRef = empty($imageRef) ? null : $imageRef;
        $images = [];

        ApiHelper::setOpenAiKey();

        if ($imageRef) {
            $service = app(CreateImageEditService::class)
                ->setImages(is_array($imageRef) ? $imageRef : [$imageRef])
                ->setModel($model->value)
                ->setPrompt($prompt);
        } else {
            $service = app(CreateImageService::class)
                ->setModel($model->value)
                ->setPrompt($prompt);
        }

        if (isset($data['size'])) {
            $service->setSize($data['size']);
        }

        if (isset($data['background'])) {
            $service->setBackground($data['background']);
        }

        if (isset($data['quality'])) {
            $service->setQuality($data['quality']);
        }

        $base64 = $service->generateForAi();

        if ($base64) {
            $images[] = base64_decode($base64);
        }

        return $images;
    }
}
