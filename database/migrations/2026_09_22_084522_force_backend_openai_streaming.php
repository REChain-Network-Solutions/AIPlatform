<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        DB::table('settings_two')->update(['openai_default_stream_server' => 'backend']);
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        // Frontend streaming exposed long-lived provider credentials and must not be restored.
    }
};
