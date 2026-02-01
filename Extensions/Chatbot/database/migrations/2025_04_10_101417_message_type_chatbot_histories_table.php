<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasColumn(self::$prefix . '_chatbot_histories', 'message_type')) {
            return;
        }

        Schema::table(self::$prefix . '_chatbot_histories', function (Blueprint $table) {
            $table->string('message_id')->nullable()->after('user_id');
            $table->text('media_url')->nullable();
            $table->string('message_type')->nullable()->after('type')->default('text');
            $table->string('content_type')->nullable()->after('message_type')->default('text');
        });
    }

    public function down(): void
    {
        Schema::table(self::$prefix . '_chatbot_histories', function (Blueprint $table) {
            $table->dropColumn(['message_type', 'content_type']);
        });
    }
};
