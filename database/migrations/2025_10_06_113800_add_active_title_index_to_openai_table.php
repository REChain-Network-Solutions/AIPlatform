<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('openai', function (Blueprint $table) {
            $table->index(['active', 'title'], 'idx_openai_active_title');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('openai', function (Blueprint $table) {
            $table->dropIndex('idx_openai_active_title');
        });
    }
};
