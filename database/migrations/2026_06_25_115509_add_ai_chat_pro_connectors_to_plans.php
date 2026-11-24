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
        Schema::table('plans', function (Blueprint $table) {
            if (! Schema::hasColumn('plans', 'ai_chat_pro_connectors')) {
                $table->json('ai_chat_pro_connectors')->nullable()->after('phone_call_agent_seconds_limit');
            }
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (Schema::hasColumn('plans', 'ai_chat_pro_connectors')) {
                $table->dropColumn('ai_chat_pro_connectors');
            }
        });
    }
};
