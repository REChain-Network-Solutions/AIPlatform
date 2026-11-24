<?php

declare(strict_types=1);

namespace App\Services;

use App\Domains\Entity\Enums\EntityEnum;
use App\Helpers\Classes\DefaultOpenAiImageModel;
use App\Models\SettingTwo;

class GeneratorMappingService
{
    /**
     * Map an image generator identifier to its corresponding EntityEnum.
     *
     * Mirrors the resolution logic from AIController::imageOutput().
     *
     * Accepts the raw enum value (gpt-image-2.5-sunburst), the URL-safe slug
     * form (gpt-image-2__5-sunburst, as sent by the AI Image Pro cost preview),
     * and the underscore tab keys (gpt_image_2_5_sunburst).
     */
    public function mapGeneratorToEntity(string $generator, ?string $action = null): ?EntityEnum
    {
        $generator = str_replace('__', '.', $generator);

        return match ($generator) {
            'openai'             => $this->getDefaultOpenAiImageModel(),
            'stable_diffusion'   => $this->getStableDiffusionDefaultModel(),
            'gpt-image-1',
            'gpt_image_1'       => EntityEnum::GPT_IMAGE_1,
            'gpt-image-1.5',
            'gpt-image-1-5',
            'gpt_image_1_5'     => EntityEnum::GPT_IMAGE_1_5,
            'gpt-image-2',
            'gpt_image_2'       => EntityEnum::GPT_IMAGE_2,
            'gpt-image-2.5-sunburst',
            'gpt_image_2_5_sunburst' => EntityEnum::GPT_IMAGE_2_5_SUNBURST,
            'gpt-image-2.5-flare',
            'gpt_image_2_5_flare'    => EntityEnum::GPT_IMAGE_2_5_FLARE,
            'flux-pro'               => $this->getDefaultFalAiModel(),
            'flux-pro-kontext',
            'flux-pro/kontext/text-to-image' => EntityEnum::FLUX_PRO_KONTEXT_TEXT_TO_IMAGE,
            'flux-pro/kontext'               => $this->resolveFluxProKontext($action),
            'nano-banana/edit'               => EntityEnum::NANO_BANANA_EDIT,
            'nano-banana-pro/edit'           => EntityEnum::NANO_BANANA_PRO_EDIT,
            'nano-banana-2/edit'             => EntityEnum::NANO_BANANA_2_EDIT,
            'ideogram',
            'ideogram-v2'                 => EntityEnum::IDEOGRAM,
            'nano-banana'                 => EntityEnum::NANO_BANANA,
            'nano-banana-pro'             => EntityEnum::NANO_BANANA_PRO,
            'nano-banana-2'               => EntityEnum::NANO_BANANA_2,
            'flux-2-flex'                 => EntityEnum::FLUX_2_FLEX,
            'flux-2-flex/edit'            => EntityEnum::FLUX_2_FLEX_EDIT,
            'flux/schnell'                => EntityEnum::FLUX_SCHNELL,
            'flux-pro/v1.1'               => EntityEnum::FLUX_PRO_1_1,
            'flux-realism'                => EntityEnum::FLUX_REALISM,
            'imagen4'                     => EntityEnum::IMAGEN_4,
            'seedream/v4/text-to-image'   => EntityEnum::SEEDREAM_4,
            'xai/grok-imagine-image'      => EntityEnum::GROK_IMAGINE_IMAGE,
            'xai/grok-imagine-image/edit' => EntityEnum::GROK_IMAGINE_IMAGE_EDIT,
            EntityEnum::MIDJOURNEY->value => EntityEnum::MIDJOURNEY,
            default                       => EntityEnum::tryFrom($generator),
        };
    }

    /**
     * Mirrors AdvancedImageController::getModel() logic for flux-pro/kontext.
     */
    private function resolveFluxProKontext(?string $action): EntityEnum
    {
        $multiTools = ['cleanup', 'image_relight', 'style_transfer'];

        if ($action && in_array($action, $multiTools, true)) {
            return EntityEnum::FLUX_PRO_KONTEXT_MAX_MULTI;
        }

        return EntityEnum::FLUX_PRO_KONTEXT;
    }

    private function getDefaultOpenAiImageModel(): EntityEnum
    {
        return DefaultOpenAiImageModel::resolve();
    }

    private function getStableDiffusionDefaultModel(): EntityEnum
    {
        $settingsTwo = SettingTwo::getCache();

        return EntityEnum::fromSlug($settingsTwo?->stablediffusion_default_model) ?? EntityEnum::SD_3_5_LARGE;
    }

    private function getDefaultFalAiModel(): EntityEnum
    {
        return EntityEnum::fromSlug(setting('fal_ai_default_model', EntityEnum::FLUX_PRO->value)) ?? EntityEnum::FLUX_PRO;
    }
}
