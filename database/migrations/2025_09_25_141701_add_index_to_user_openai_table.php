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
        Schema::table('user_openai', function (Blueprint $table) {
            $table->index(['user_id', 'updated_at'], 'idx_user_id_updated_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('user_openai', function (Blueprint $table) {
            $table->dropIndex('idx_user_id_updated_at');
        });
    }
};
