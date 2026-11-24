<?php

use App\Domains\Entity\Models\Entity;
use App\Helpers\Classes\EntityRemover;
use App\Models\OpenAIGenerator;
use App\Models\Plan;
use App\Models\SettingTwo;
use Illuminate\Database\Migrations\Migration;

/**
 * Retires the image and video models whose provider APIs are switched off:
 *
 * - dall-e-2 / dall-e-3           removed from the OpenAI API on 2026-05-12
 * - stable-diffusion-v1-6         Stability discontinued the SD1.6 API on 2025-07-24
 * - image-to-video                Stability discontinued Stable Video on 2025-07-24
 * - sd3 / sd3-turbo / sd3-medium / sd3-large / sd3-large-turbo
 *                                 SD3.0 ids, auto-upgraded to sd3.5-* on 2025-04-17
 *
 * Order matters: every stored default is repointed BEFORE EntityRemover runs.
 * EntityRemover::removeEntity() refuses to run while the case is still in EntityEnum,
 * so this migration must ship after the enum change; and it only nulls settings keys
 * containing '_default_', which never covers the settings_two.dalle column.
 */
return new class extends Migration
{
    private array $openAiModels = ['dall-e-2', 'dall-e-3'];

    private array $stableDiffusionModels = [
        'stable-diffusion-v1-6',
        'image-to-video',
        'sd3',
        'sd3-turbo',
        'sd3-medium',
        'sd3-large',
        'sd3-large-turbo',
    ];

    public function up(): void
    {
        $removed = [...$this->openAiModels, ...$this->stableDiffusionModels];

        try {
            $this->repointStoredDefaults();

            foreach ($removed as $model) {
                EntityRemover::removeEntity($model);
            }
        } catch (Throwable $exception) {
        }

        // Guaranteed floor: the catch above swallows anything EntityRemover throws,
        // and an orphan entities row makes Entity::$casts['key'] blow up on render.
        Entity::query()->whereIn('key', $removed)->delete();
    }

    public function down(): void {}

    private function repointStoredDefaults(): void
    {
        $settingsTwo = SettingTwo::query()->first();

        if ($settingsTwo) {
            // 'dalle2'/'dalle3' are the pre-2024_09_24_214000 setting format.
            if (in_array($settingsTwo->dalle, [...$this->openAiModels, 'dalle2', 'dalle3'], true)) {
                $settingsTwo->update(['dalle' => 'gpt-image-2']);
            }

            foreach (['stablediffusion_default_model', 'stable_diffusion_default_model'] as $column) {
                if (in_array($settingsTwo->{$column}, $this->stableDiffusionModels, true)) {
                    $settingsTwo->update([$column => 'stable-diffusion-xl-1024-v1-0']);
                }
            }

            $settingsTwo->save();
            SettingTwo::forgetCache();
        }

        Plan::query()->whereIn('default_ai_model', $this->openAiModels)
            ->update(['default_ai_model' => 'gpt-image-2']);

        Plan::query()->whereIn('default_ai_model', $this->stableDiffusionModels)
            ->update(['default_ai_model' => 'stable-diffusion-xl-1024-v1-0']);

        // The Stable Diffusion video generator page is gone; the row itself stays,
        // because FashionStudio and AIPhotoshoot file their Veo videos under it.
        OpenAIGenerator::query()->where('slug', 'ai_video')->update(['active' => 0]);
    }
};
