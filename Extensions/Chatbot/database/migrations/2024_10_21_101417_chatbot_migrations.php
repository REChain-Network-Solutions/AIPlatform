<?php

use App\Extensions\Chatbot\System\Enums\ColorModeEnum;
use App\Extensions\Chatbot\System\Enums\PositionEnum;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasTable(self::$prefix . '_chatbots')) {
            return;
        }

        Schema::create(self::$prefix . '_chatbots', function (Blueprint $table) {
            $table->id();
            $table->uuid('uuid')->unique();
            $table->bigInteger('user_id');
            $table->string('title');
            $table->text('bubble_message')->nullable();
            $table->text('welcome_message')->nullable();
            $table->text('instructions')->nullable();
            $table->boolean('do_not_go_beyond_instructions')->default(false);
            $table->string('language')->nullable();
            $table->string('ai_model');
            $table->string('ai_embedding_model');
            $table->bigInteger('limit_per_minute')->default(2000);
            $table->boolean('show_pre_defined_questions')->default(false);
            $table->json('pre_defined_questions')->nullable();
            // customizations start
            $table->longText('logo')->nullable()->comment('base 64');
            $table->string('avatar')->nullable();
            $table->string('trigger_avatar_size')->nullable();
            $table->string('trigger_background')->nullable();
            $table->string('trigger_foreground')->nullable();
            $table->string('color_mode')->default(ColorModeEnum::none->value);
            $table->string('color')->nullable()->default('#017BE5');
            $table->boolean('show_logo')->default(true);
            $table->boolean('show_date_and_time')->default(true);
            $table->boolean('show_average_response_time')->default(true);
            $table->string('position')->default(PositionEnum::right->value);
            // customizations end
            $table->boolean('active')->default(true);
            $table->timestamps();
        });

        Schema::create(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            $table->id();
            $table->foreignId('chatbot_id')->constrained(self::$prefix . '_chatbots')->cascadeOnDelete();
            $table->string('session_id');
            $table->timestamps();
        });

        Schema::create(self::$prefix . '_chatbot_histories', function (Blueprint $table) {
            $table->id();
            $table->foreignId('chatbot_id')->constrained(self::$prefix . '_chatbots')->cascadeOnDelete();
            $table->foreignId('conversation_id')->constrained(self::$prefix . '_chatbot_conversations')->cascadeOnDelete();
            $table->string('model')->nullable();
            $table->string('role')->nullable();
            $table->text('message');
            $table->timestamp('created_at')->useCurrent();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists(self::$prefix . '_chatbot_histories');
        Schema::dropIfExists(self::$prefix . '_chatbot_conversations');
        Schema::dropIfExists(self::$prefix . '_chatbots');
    }
};
