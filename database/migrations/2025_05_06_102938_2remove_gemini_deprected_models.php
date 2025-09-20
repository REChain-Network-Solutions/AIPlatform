<?php

use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        DB::table('migrations')
            ->where('migration', '2025_05_05_102938_remove_gemini_deprected_models')
            ->delete();

        $modelsToBeRemoved = [
            'gemini-pro-vision',
            'gemini-pro',
            'gemini-1.5-pro-latest',
            'gemini-1__5-pro-latest',
        ];
        $default = setting('gemini_default_model');
        if (in_array($default, $modelsToBeRemoved, true)) {
            $newDefault = 'gemini-1.5-flash';
            setting([
                'gemini_default_model' => $newDefault,
            ])->save();
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
