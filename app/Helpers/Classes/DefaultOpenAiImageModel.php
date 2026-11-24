<?php

declare(strict_types=1);

namespace App\Helpers\Classes;

use App\Domains\Entity\Enums\EntityEnum;
use App\Models\SettingTwo;

/**
 * Single source of truth for "which OpenAI image model is the site default".
 *
 * Reads settings_two.dalle, which historically stored the legacy slugs
 * 'dalle2'/'dalle3' (pre 2024_09_24_214000), then 'dall-e-2'/'dall-e-3'.
 * Both DALL-E models were removed from the OpenAI API on 2026-05-12 and from
 * EntityEnum; anything unrecognised degrades to gpt-image-2.
 *
 * Uses tryFrom(), never fromSlug(): fromSlug() throws on an unknown slug and
 * additionally triggers EntityRemover, which would 500 every install whose
 * settings_two.dalle still holds a retired id.
 */
class DefaultOpenAiImageModel
{
    public const FALLBACK = EntityEnum::GPT_IMAGE_2;

    public static function resolve(?SettingTwo $settingTwo = null): EntityEnum
    {
        return self::fromValue(($settingTwo ?? SettingTwo::getCache())?->dalle);
    }

    public static function fromValue(?string $value): EntityEnum
    {
        return match ($value) {
            null, '', 'dalle2', 'dalle3', 'dall-e-2', 'dall-e-3' => self::FALLBACK,
            default => EntityEnum::tryFrom(str_replace('__', '.', $value)) ?? self::FALLBACK,
        };
    }

    public static function slug(?SettingTwo $settingTwo = null): string
    {
        return self::resolve($settingTwo)->slug();
    }
}
