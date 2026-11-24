<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasColumn('plans', 'video_dubbing_seconds_limit')) {
            return;
        }

        Schema::table('plans', function (Blueprint $table) {
            $table->integer('video_dubbing_seconds_limit')->default(20);
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            $table->dropColumn('video_dubbing_seconds_limit');
        });
    }
};
