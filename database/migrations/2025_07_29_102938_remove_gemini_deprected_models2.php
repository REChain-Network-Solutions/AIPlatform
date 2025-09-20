<?php

use App\Helpers\Classes\EntityRemover;
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
                'gemini-2.5-flash-preview-04-17',
                'gemini-2.5-pro-exp-03-25',
                'gemini-2__5-flash-preview-04-17',
                'gemini-2__5-pro-exp-03-25',
            ];

            $default = setting('gemini_default_model');
            if (in_array($default, $modelsToBeRemoved, true)) {
                $newDefault = 'gemini-1.5-flash';
                setting([
                    'gemini_default_model' => $newDefault,
                ])->save();
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
