<?php

use App\Extensions\Chatbot\System\Enums\InteractionType;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        Schema::table(self::$prefix . '_chatbots', function (Blueprint $table) {
            if (Schema::hasColumn(self::$prefix . '_chatbots', 'interaction_type')) {
                return;
            }

            $table->string('interaction_type')->nullable()
                ->default(InteractionType::AUTOMATIC_RESPONSE->value)
                ->after('user_id');
            $table->text('connect_message')->nullable()->after('welcome_message');
        });

        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            if (! Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'connect_agent_at')) {
                $table->timestamp('connect_agent_at')->nullable()->after('session_id');
            }
        });

        Schema::table(self::$prefix . '_chatbot_histories', function (Blueprint $table) {
            if (Schema::hasColumn(self::$prefix . '_chatbot_histories', 'user_id')) {
                return;
            }

            $table->bigInteger('user_id')->nullable()->after('conversation_id');
            $table->string('type')->nullable()->after('message')->default('default');
            $table->timestamp('read_at')->nullable()->after('type');
        });
    }

    public function down(): void
    {
        Schema::table(self::$prefix . '_chatbot_conversations', function (Blueprint $table) {
            if (! Schema::hasColumn(self::$prefix . '_chatbot_conversations', 'connect_agent_at')) {
                return;
            }

            $table->dropColumn('connect_agent_at');
        });

        Schema::table(self::$prefix . '_chatbot_histories', function (Blueprint $table) {
            if (! Schema::hasColumn(self::$prefix . '_chatbot_histories', 'user_id')) {
                return;
            }
            $table->dropColumn('user_id');
            $table->dropColumn('type');
            $table->dropColumn('read_at');
        });
    }
};
