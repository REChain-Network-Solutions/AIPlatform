<?php

use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        setting([
            'ai_realtime_image' => '1',
        ]);

        setting()->save();
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void {}
};
