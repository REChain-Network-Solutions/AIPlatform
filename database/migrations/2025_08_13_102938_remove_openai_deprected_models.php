<?php

use App\Helpers\Classes\EntityRemover;
use App\Models\Setting;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        try {
            $modelsToBeRemoved = [
                'gpt-3.5-turbo-16k', 'gpt-4-vision-preview', 'gpt-4.5-preview',
                'gpt-3__5-turbo-16k', 'gpt-4__5-preview',
            ];

            $settings = Setting::getCache();
            $default = $settings?->openai_default_model;
            if (in_array($default, $modelsToBeRemoved, true)) {
                $newDefault = 'gpt-4o';
                $settings->update(['openai_default_model' => $newDefault]);
                Setting::forgetCache();
            }

            foreach ($modelsToBeRemoved as $model) {
                EntityRemover::removeEntity($model);
            }
        } catch (\Throwable $exception) {
        }
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        //
    }
};
