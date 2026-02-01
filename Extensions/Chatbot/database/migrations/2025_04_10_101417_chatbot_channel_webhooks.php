<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasTable(self::$prefix . '_chatbot_channel_webhooks')) {
            return;
        }

        Schema::create(self::$prefix . '_chatbot_channel_webhooks', function (Blueprint $table) {
            $table->id();
            $table->bigInteger('chatbot_id')->nullable();
            $table->bigInteger('chatbot_channel_id')->nullable();
            $table->json('payload')->nullable();
            $table->timestamp('created_at')->nullable()->useCurrent();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists(self::$prefix . '_chatbot_channel_webhooks');
    }
};
