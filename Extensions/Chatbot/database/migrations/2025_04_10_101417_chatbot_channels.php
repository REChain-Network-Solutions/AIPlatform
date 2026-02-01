<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasTable(self::$prefix . '_chatbot_channels')) {
            return;
        }

        Schema::create(self::$prefix . '_chatbot_channels', function (Blueprint $table) {
            $table->id();
            $table->bigInteger('user_id')->nullable();
            $table->bigInteger('chatbot_id')->nullable();
            $table->string('channel', 32)->nullable();
            $table->json('credentials')->nullable();
            $table->json('payload')->nullable();
            $table->timestamp('connected_at')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists(self::$prefix . '_chatbot_channels');
    }
};
