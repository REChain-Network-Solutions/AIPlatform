<?php

declare(strict_types=1);

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasColumn('plans', 'ugc_videos_limit')) {
            return;
        }

        Schema::table('plans', function (Blueprint $table) {
            $table->integer('ugc_videos_limit')->default(-1);
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            $table->dropColumn('ugc_videos_limit');
        });
    }
};
