<?php

declare(strict_types=1);

use App\Domains\Entity\Enums\EntityEnum;
use App\Extensions\AIPhotoshoot\System\Services\AIPhotoshootImageModelRegistry;

test('falls back to nano banana pro when no setting is configured', function () {
    expect(AIPhotoshootImageModelRegistry::getDefaultModel())
        ->toBe(EntityEnum::NANO_BANANA_PRO);
});

test('maps fal nano-banana-pro to fal provider', function () {
    expect(AIPhotoshootImageModelRegistry::getProviderFor(EntityEnum::NANO_BANANA_PRO))
        ->toBe(AIPhotoshootImageModelRegistry::PROVIDER_FAL);
});

test('maps gpt-image-* to openai provider', function () {
    expect(AIPhotoshootImageModelRegistry::getProviderFor(EntityEnum::GPT_IMAGE_1))
        ->toBe(AIPhotoshootImageModelRegistry::PROVIDER_OPENAI)
        ->and(AIPhotoshootImageModelRegistry::getProviderFor(EntityEnum::GPT_IMAGE_1_5))
        ->toBe(AIPhotoshootImageModelRegistry::PROVIDER_OPENAI)
        ->and(AIPhotoshootImageModelRegistry::getProviderFor(EntityEnum::GPT_IMAGE_2))
        ->toBe(AIPhotoshootImageModelRegistry::PROVIDER_OPENAI);
});

test('exposes admin-selectable models', function () {
    $models = AIPhotoshootImageModelRegistry::getModelsForAdminSelect();

    expect($models)
        ->toHaveKey(EntityEnum::NANO_BANANA_PRO->value)
        ->toHaveKey(EntityEnum::GPT_IMAGE_2->value);
});

test('snaps user resolution+ratio to closest valid OpenAI size', function () {
    expect(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', '1:1'))->toBe('1024x1024')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', '16:9'))->toBe('1536x1024')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', '9:16'))->toBe('1024x1536')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('2K', '16:9'))->toBe('2048x1152')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('4K', '16:9'))->toBe('3840x2160')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('2K', '9:16'))->toBe('2048x2048')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', 'auto'))->toBe('auto')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', 'invalid'))->toBe('1024x1024');
});

test('passes through OpenAI size strings unchanged', function () {
    expect(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', '1024x1024'))->toBe('1024x1024')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', '2048x2048'))->toBe('2048x2048')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('4K', '3840x2160'))->toBe('3840x2160')
        ->and(AIPhotoshootImageModelRegistry::resolveOpenAISizeForUser('1K', '2160x3840'))->toBe('2160x3840');
});

test('clamps invalid quality and size settings to defaults', function () {
    expect(AIPhotoshootImageModelRegistry::getOpenAIQuality())
        ->toBe(AIPhotoshootImageModelRegistry::OPENAI_QUALITY_DEFAULT)
        ->and(AIPhotoshootImageModelRegistry::getOpenAISize())
        ->toBe(AIPhotoshootImageModelRegistry::OPENAI_SIZE_DEFAULT);
});

test('exposes only ratios FAL can produce exactly for fal provider', function () {
    $options = AIPhotoshootImageModelRegistry::getRatioOptionsForProvider(
        AIPhotoshootImageModelRegistry::PROVIDER_FAL
    );

    expect(array_keys($options))
        ->toContain('auto', '1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3', '5:4', '4:5', '21:9');
});

test('exposes OpenAI size strings as ratio options for openai provider', function () {
    $options = AIPhotoshootImageModelRegistry::getRatioOptionsForProvider(
        AIPhotoshootImageModelRegistry::PROVIDER_OPENAI
    );

    expect(array_keys($options))
        ->toEqualCanonicalizing([
            'auto',
            '1024x1024',
            '2048x2048',
            '1536x1024',
            '2048x1152',
            '3840x2160',
            '1024x1536',
            '2160x3840',
        ]);
});
