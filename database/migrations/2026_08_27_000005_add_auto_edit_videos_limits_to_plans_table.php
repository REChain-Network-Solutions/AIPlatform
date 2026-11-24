<?php

declare(strict_types=1);

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * `plans` is a core table, so this migration lives in the application rather
 * than inside the Auto Edit Videos extension.
 *
 * -1 means unlimited and 0 means the feature is off for that plan, matching
 * every other per-plan allowance in this application.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (! Schema::hasColumn('plans', 'auto_edit_videos_seconds_limit')) {
                $table->integer('auto_edit_videos_seconds_limit')->default(-1);
            }

            if (! Schema::hasColumn('plans', 'auto_edit_videos_renders_limit')) {
                $table->integer('auto_edit_videos_renders_limit')->default(-1);
            }
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (Schema::hasColumn('plans', 'auto_edit_videos_seconds_limit')) {
                $table->dropColumn('auto_edit_videos_seconds_limit');
            }

            if (Schema::hasColumn('plans', 'auto_edit_videos_renders_limit')) {
                $table->dropColumn('auto_edit_videos_renders_limit');
            }
        });
    }
};
