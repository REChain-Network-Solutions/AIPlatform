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
        Schema::table('user_openai_chat_messages', function (Blueprint $table) {
            if (! Schema::hasColumn('user_openai_chat_messages', 'used_skills')) {
                $table->json('used_skills')->nullable();
            }
        });
    }

    public function down(): void
    {
        Schema::table('user_openai_chat_messages', function (Blueprint $table) {
            $table->dropColumn('used_skills');
        });
    }
};
