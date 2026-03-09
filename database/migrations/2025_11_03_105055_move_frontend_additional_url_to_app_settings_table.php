<?php

use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        $frontend_additional_url = \App\Models\Setting::first()->frontend_additional_url ?? null;
        $frontend_type = match (true) {
            str_ends_with($frontend_additional_url, '/chat')         => 'ai-chat-pro',
            str_ends_with($frontend_additional_url, '/ai-image-pro') => 'ai-image-pro',
            default                                                  => 'default',
        };
        setting([
            'frontend_additional_url_type' => $frontend_type,
            'frontend_additional_url'      => $frontend_additional_url,
        ])->save();
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void {}
};
