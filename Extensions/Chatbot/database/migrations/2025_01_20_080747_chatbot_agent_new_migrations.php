<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static string $prefix = 'ext';

    public function up(): void
    {
        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            if (! Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'ip_address')) {
                $table->ipAddress()->nullable()->after('id');
            }

            if (! Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'conversation_name')) {
                $table->string('conversation_name')->nullable()->default('Anonymous User')->after('ip_address');
            }
        });
    }

    public function down(): void
    {
        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            if (Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'ip_address')) {
                $table->dropColumn('ip_address');
                $table->dropColumn('conversation_name');
            }
        });
    }
};
