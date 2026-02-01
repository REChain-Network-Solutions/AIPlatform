<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasColumn('user_openai', 'is_advanced_image')) {
            return;
        }

        Schema::table('user_openai', function (Blueprint $table) {
            $table->boolean('is_advanced_image')->default(false)->after('request_id');
        });
    }

    public function down(): void {}
};
