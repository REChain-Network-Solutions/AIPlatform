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
            if (! Schema::hasColumn('plans', 'chatbot_limit')) {
                $table->integer('chatbot_limit')
                    ->nullable()
                    ->after('max_subscribe');
            }
            if (! Schema::hasColumn('plans', 'chatbot_channels')) {
                $table->json('chatbot_channels')
                    ->nullable()
                    ->after('chatbot_limit');
            }
            if (! Schema::hasColumn('plans', 'chatbot_human_agent')) {
                $table->boolean('chatbot_human_agent')
                    ->default(true)
                    ->after('chatbot_channels');
            }
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (Schema::hasColumn('plans', 'chatbot_limit')) {
                $table->dropColumn('chatbot_limit');
            }
            if (Schema::hasColumn('plans', 'chatbot_channels')) {
                $table->dropColumn('chatbot_channels');
            }
            if (Schema::hasColumn('plans', 'chatbot_human_agent')) {
                $table->dropColumn('chatbot_human_agent');
            }
        });
    }
};
