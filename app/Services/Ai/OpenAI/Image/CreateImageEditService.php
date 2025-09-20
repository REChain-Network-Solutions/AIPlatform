<?php

namespace App\Services\Ai\OpenAI\Image;

use App\Helpers\Classes\Helper;
use Illuminate\Support\Arr;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use OpenAI;

class CreateImageEditService
{
    private string $generateURL = 'https://api.openai.com/v1/images/edits';

    /**
     * Supported models: dall-e-2, gpt-image-1
     */
    private string $model = 'gpt-image-1';

    private string $prompt = '';

    private string $size = '1024x1024';

    /**
     * dall-e-2: 'standard'
     * gpt-image-1: 'auto', 'high', 'low'
     */
    private string $quality = 'auto';

    private ?string $mask = null;

    private array $images = [];

    public string $output_format = 'png';

    public function generateForAi(): ?string
    {
        $client = OpenAI::client(Helper::setOpenAiKey());

        $response = $client
            ->images()
            ->edit($this->requestData());

        if (! isset($response['created'])) {
            return null;
        }

        if ($response['created'] && $response['data']) {
            $image = Arr::first($response['data']);
            if (isset($image['b64_json'])) {
                return $image['b64_json'];
            }
        }

        return null;
    }

    public function generate(): array
    {
        $client = OpenAI::client(Helper::setOpenAiKey());

        $response = $client
            ->images()
            ->edit($this->requestData());

        if (! isset($response['created'])) {
            return [
                'status'  => false,
                'message' => trans('AI model not available'),
            ];
        }

        if ($response['created'] && $response['data']) {

            $image = Arr::first($response['data']);

            if (isset($image['b64_json'])) {
                $base64Image = $image['b64_json'];

                $imageName = Str::uuid()->toString() . '.' . $this->getOutputFormat();

                Storage::disk('uploads')->put($imageName, base64_decode($base64Image));

                return [
                    'status'  => true,
                    'message' => trans('Image generated successfully'),
                    'path'    => '/uploads/' . $imageName,
                ];
            }
        }

        return [
            'status'  => false,
            'message' => 'Image generation failed. Please try again later.',
        ];
    }

    public function requestData(): array
    {
        $images = $this->getImages();

        $dataImage = [];

        if (count($images) > 1) {
            foreach ($this->getImages() as $image) {
                $dataImage[] = fopen(public_path($image), 'r');
            }
        } else {
            $dataImage = fopen(public_path(Arr::first($images)), 'r');
        }

        $form = [
            'image'           => $dataImage,
            'model'           => $this->getModel(),
            'prompt'          => $this->getPrompt(),
            'n'               => 1,
            'size'            => $this->getSize(),
        ];

        if ($this->mask && file_exists(public_path($this->mask))) {
            $form['mask'] = fopen(public_path($this->mask), 'r');
        }

        return $form;
    }

    public function setModel(string $model): self
    {
        $this->model = $model;

        return $this;
    }

    public function getModel(): string
    {
        return $this->model;
    }

    public function setPrompt(string $prompt): self
    {
        $this->prompt = $prompt;

        return $this;
    }

    public function getPrompt(): string
    {
        return $this->prompt;
    }

    public function setSize(string $size): self
    {
        $this->size = $size;

        return $this;
    }

    public function getSize(): string
    {
        return $this->size;
    }

    public function setQuality(string $quality): self
    {
        $this->quality = $quality;

        return $this;
    }

    public function getQuality(): string
    {
        return $this->quality;
    }

    public function setMask(?string $mask): self
    {
        $this->mask = $mask;

        return $this;
    }

    public function getMask()
    {
        return $this->mask;
    }

    public function setImages(array $images): self
    {
        $this->images = $images;

        return $this;
    }

    public function getImages(): array
    {
        return $this->images;
    }

    public function setOutputFormat(string $output_format): self
    {
        $this->output_format = $output_format;

        return $this;
    }

    public function getOutputFormat(): string
    {
        return $this->output_format;
    }
}
