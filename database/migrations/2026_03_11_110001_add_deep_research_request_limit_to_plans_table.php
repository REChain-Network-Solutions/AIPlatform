<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasColumn('plans', 'deep_research_request_limit')) {
            return;
        }

        Schema::table('plans', function (Blueprint $table) {
            $table->integer('deep_research_request_limit')->default(5);
        });
    }

    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            $table->dropColumn('deep_research_request_limit');
        });
    }
};
