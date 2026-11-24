<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (! Schema::hasColumn('plans', 'ai_captions_access')) {
                $table->boolean('ai_captions_access')->default(true);
            }
            if (! Schema::hasColumn('plans', 'ai_captions_minutes')) {
                $table->integer('ai_captions_minutes')->default(30);
            }
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            if (Schema::hasColumn('plans', 'ai_captions_access')) {
                $table->dropColumn('ai_captions_access');
            }
            if (Schema::hasColumn('plans', 'ai_captions_minutes')) {
                $table->dropColumn('ai_captions_minutes');
            }
        });
    }
};
