<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'chatbot_channel_id')) {
            return;
        }

        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->bigInteger('chatbot_channel_id')->default(0)->nullable()->after('id');
            $table->string('customer_channel_id')->nullable()->after('chatbot_channel_id');
            $table->json('customer_payload')->nullable()->after('connect_agent_at');
            $table->boolean('is_showed_on_history')->default(true);
        });
    }

    public function down(): void
    {
        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->dropColumn(['chatbot_channel_id', 'customer_channel_id']);
        });
    }
};
