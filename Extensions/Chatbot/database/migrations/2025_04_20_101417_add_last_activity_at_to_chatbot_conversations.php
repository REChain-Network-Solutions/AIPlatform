<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'last_activity_at')) {
            return;
        }

        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->timestamp('last_activity_at')->nullable()->after('updated_at');
        });
    }

    public function down(): void
    {
        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->dropColumn(['last_activity_at']);
        });
    }
};
