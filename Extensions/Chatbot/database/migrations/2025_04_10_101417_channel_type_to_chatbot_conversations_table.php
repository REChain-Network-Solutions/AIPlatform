<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'chatbot_channel')) {
            return;
        }

        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->string('chatbot_channel')->nullable()->after('id')->default('frame');
        });
    }

    public function down(): void
    {
        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->dropColumn('chatbot_channel');
        });
    }
};
