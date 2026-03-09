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
        Schema::table('recent_search_keys', function (Blueprint $table) {
            $table->index(['user_id', 'keyword'], 'recent_search_keys_user_keyword_index');
            $table->index(['user_id', 'created_at'], 'recent_search_keys_user_created_index');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('recent_search_keys', function (Blueprint $table) {
            $table->dropIndex('recent_search_keys_user_keyword_index');
            $table->dropIndex('recent_search_keys_user_created_index');
        });
    }
};
