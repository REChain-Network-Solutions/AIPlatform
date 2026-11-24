<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (! Schema::hasColumn('plans', 'ai_agent_workflow_limit')) {
                $table->integer('ai_agent_workflow_limit')->nullable()->after('deep_research_request_limit');
            }

            if (! Schema::hasColumn('plans', 'ai_agent_channel_limit')) {
                $table->integer('ai_agent_channel_limit')->nullable()->after('ai_agent_workflow_limit');
            }

            if (! Schema::hasColumn('plans', 'ai_agent_message_limit')) {
                $table->integer('ai_agent_message_limit')->nullable()->after('ai_agent_channel_limit');
            }

            if (! Schema::hasColumn('plans', 'ai_agent_memory_limit')) {
                $table->integer('ai_agent_memory_limit')->nullable()->after('ai_agent_message_limit');
            }
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            $table->dropColumn([
                'ai_agent_workflow_limit',
                'ai_agent_channel_limit',
                'ai_agent_message_limit',
                'ai_agent_memory_limit',
            ]);
        });
    }
};
