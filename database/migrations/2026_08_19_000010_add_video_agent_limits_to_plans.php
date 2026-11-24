<?php

declare(strict_types=1);

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (! Schema::hasColumn('plans', 'video_agent_seconds_limit')) {
                $table->integer('video_agent_seconds_limit')->default(-1);
            }

            if (! Schema::hasColumn('plans', 'video_agent_videos_limit')) {
                $table->integer('video_agent_videos_limit')->default(-1);
            }
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (Schema::hasColumn('plans', 'video_agent_seconds_limit')) {
                $table->dropColumn('video_agent_seconds_limit');
            }

            if (Schema::hasColumn('plans', 'video_agent_videos_limit')) {
                $table->dropColumn('video_agent_videos_limit');
            }
        });
    }
};
